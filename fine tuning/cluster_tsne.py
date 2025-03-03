import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from transformers import BertConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import pickle

# Set logging level and format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    device = torch.device("cuda:1")
except Exception as e:
    device = torch.device("cuda:0")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

###################################
# Custom BERT architecture (consistent with pre-training)
###################################
from transformers.models.bert.modeling_bert import BertModel, BertEncoder, BertLayer, BertSelfAttention

class CustomBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            bias_matrix_chunk=None,
            bias_coef=None
    ):
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
        else:
            past_key_value = None

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Align the dynamically generated bias_matrix_chunk with attention_mask based on chunk size.
        # At this time, seq_len should match bias_matrix_chunk.
        # bias_matrix_chunk should be of shape [seq_len, seq_len].
        if bias_matrix_chunk is not None and bias_coef is not None:
            batch_size, num_heads, seq_len, _ = attention_scores.size()
            bias = bias_matrix_chunk.unsqueeze(0).unsqueeze(0) * bias_coef
            bias = bias.expand(batch_size, num_heads, seq_len, seq_len)
            attention_scores = attention_scores + bias

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,)
        if output_attentions:
            outputs = outputs + (attention_probs,)
        return outputs  # (context_layer, attention_probs)

class CustomBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = CustomBertSelfAttention(config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.pruned_heads = set()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            **kwargs
    ):
        self_outputs = self.self(hidden_states, attention_mask,
                                 output_attentions=kwargs.get('output_attentions', False),
                                 bias_matrix_chunk=kwargs.get('bias_matrix_chunk', None),
                                 bias_coef=kwargs.get('bias_coef', None))
        attention_output = self.output(self_outputs[0])
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)
        outputs = (attention_output,)
        if kwargs.get('output_attentions', False):
            outputs = outputs + (self_outputs[1],)
        return outputs

class CustomBertLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = CustomBertAttention(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            **kwargs
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=kwargs.get('output_attentions', False),
            bias_matrix_chunk=kwargs.get("bias_matrix_chunk", None),
            bias_coef=kwargs.get("bias_coef", None)
        )
        attention_output = self_attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,)
        if kwargs.get('output_attentions', False):
            outputs = outputs + (self_attention_outputs[1],)
        return outputs

class CustomBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([CustomBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            **kwargs
    ):
        all_attentions = [] if kwargs.get('output_attentions', False) else None

        for layer_module in self.layer:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions=kwargs.get('output_attentions', False),
                bias_matrix_chunk=kwargs.get("bias_matrix_chunk", None),
                bias_coef=kwargs.get("bias_coef", None)
            )
            hidden_states = layer_outputs[0]
            if kwargs.get('output_attentions', False):
                all_attentions.append(layer_outputs[1])

        outputs = {
            'last_hidden_state': hidden_states
        }
        if kwargs.get('output_attentions', False):
            outputs['attentions'] = all_attentions
        return outputs

class CustomBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = CustomBertEncoder(config)
        self.init_weights()

    def forward(
            self,
            inputs_embeds=None,
            attention_mask=None,
            bias_matrix_chunk=None,
            bias_coef=None,
            output_attentions=False
    ):
        if inputs_embeds is None:
            raise ValueError("inputs_embeds is required")

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, inputs_embeds.size()[:-1],
                                                                   device=inputs_embeds.device)
        encoder_outputs = self.encoder(
            hidden_states=inputs_embeds,
            attention_mask=extended_attention_mask,
            bias_matrix_chunk=bias_matrix_chunk,
            bias_coef=bias_coef,
            output_attentions=output_attentions
        )
        return encoder_outputs

class ProteinBERTModel(nn.Module):
    def __init__(self, feature_dim, hidden_size, num_layers, num_proteins, features_tensor, bias_matrix,
                 chunk_size=512):
        super(ProteinBERTModel, self).__init__()
        self.chunk_size = chunk_size
        config = BertConfig(
            vocab_size=chunk_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=chunk_size + 1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=False,
        )
        self.bert = CustomBertModel(config)
        self.feature_embedding = nn.Linear(feature_dim, hidden_size)
        self.expression_embedding = nn.Linear(1, hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.register_buffer('features_tensor', features_tensor)
        self.register_buffer('bias_matrix_full', bias_matrix)
        self.bias_coef = nn.Parameter(torch.tensor(1.0))

    def forward_one_chunk_batch(self, feature_embeds_chunk, expr_embeds_chunk, bias_matrix_chunk, attention_mask_chunk,
                                output_attentions=False):
        embeddings = feature_embeds_chunk * expr_embeds_chunk
        B, chunk_len, H = embeddings.shape
        cls_tokens = self.cls_token.expand(B, 1, H)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)  # [B, chunk_len+1, H]
        attention_mask_chunk = torch.cat(
            (torch.ones(B, 1, device=attention_mask_chunk.device), attention_mask_chunk), dim=1
        )  # [B, chunk_len+1]

        outputs = self.bert(
            inputs_embeds=embeddings,
            attention_mask=attention_mask_chunk,
            bias_matrix_chunk=bias_matrix_chunk,
            bias_coef=self.bias_coef,
            output_attentions=output_attentions
        )
        sequence_output = outputs['last_hidden_state']
        cls_emb = sequence_output[:, 0, :]  # [B, H]
        attentions = outputs.get('attentions', None)
        return cls_emb, attentions

    def forward(self, expressions, output_attentions=False):
        # expressions: [B, num_proteins]
        batch_size, num_proteins = expressions.size()
        feature_embeds = self.feature_embedding(self.features_tensor)  # [num_proteins, H]
        feature_embeds = feature_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_proteins, H]
        expression_embeds = self.expression_embedding(expressions.unsqueeze(-1))  # [B, num_proteins, H]

        total_chunks = (num_proteins + self.chunk_size - 1) // self.chunk_size
        cls_list = []
        attention_list = []
        seq_lengths = []
        chunk_indices = []  # To track which proteins are in which chunk
        for i in range(total_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, num_proteins)
            chunk_len = end - start

            feature_chunk = feature_embeds[:, start:end, :]
            expr_chunk = expression_embeds[:, start:end, :]

            # Dynamically create bias_matrix_chunk based on actual chunk size
            attention_mask_chunk = torch.ones((batch_size, chunk_len), dtype=torch.long, device=expressions.device)
            rows = [0] + list(range(1 + start, 1 + end))
            # bias_matrix_chunk shape should be [chunk_len+1, chunk_len+1]
            bias_matrix_chunk = self.bias_matrix_full[rows][:, rows]  # [chunk_len+1, chunk_len+1]

            cls_emb, attentions = self.forward_one_chunk_batch(feature_chunk, expr_chunk, bias_matrix_chunk,
                                                               attention_mask_chunk,
                                                               output_attentions=output_attentions)
            cls_list.append(cls_emb)
            if output_attentions:
                # Store attentions along with their corresponding protein indices
                # Each 'attentions' is a list of tensors per layer: [num_layers][batch_size, num_heads, seq_len, seq_len]
                attention_list.append((attentions, start, end))
            seq_lengths.append(chunk_len + 1)  # +1 for CLS token

        cls_all = torch.stack(cls_list, dim=0).mean(dim=0)  # [B, H]
        if output_attentions:
            aggregate_attention = torch.zeros(num_proteins, device=expressions.device)
            total_attention = torch.zeros(num_proteins, device=expressions.device)
            for attentions, start, end in attention_list:
                cls_attentions = []
                for layer_attn in attentions:
                    # layer_attn: [batch_size, num_heads, seq_len, seq_len]
                    # Extract [CLS] token's attention to all tokens (excluding [CLS] itself)
                    # Shape: [num_heads, seq_len]
                    cls_attn = layer_attn[0, :, 0, 1:]
                    # Average over heads
                    cls_attn_mean = cls_attn.mean(dim=0)  # [seq_len]
                    cls_attentions.append(cls_attn_mean)
                # Average over layers
                cls_attentions = torch.stack(cls_attentions, dim=0).mean(dim=0)  # [seq_len]
                # Assign to the global attention list
                aggregate_attention[start:end] += cls_attentions
                total_attention[start:end] += 1  # Count how many times a protein has been attended to

            total_attention = torch.clamp(total_attention, min=1)
            aggregate_attention = aggregate_attention / total_attention
            aggregate_attention = aggregate_attention.cpu().numpy()
            return cls_all, aggregate_attention
        else:
            return cls_all

class ProteinClassificationDataset(Dataset):
    def __init__(self, expression_data, eids, labels):
        self.expression_data = expression_data
        self.eids = eids
        self.labels = labels

    def __len__(self):
        return len(self.eids)

    def __getitem__(self, idx):
        eid = self.eids[idx]
        label = self.labels[idx]
        expr = self.expression_data.loc[eid].values.astype(np.float32)
        return torch.tensor(expr), label, eid  # Return eid

class ClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def extract_disease_labels(label_files, expression_data_index):
    disease_eids = {}
    for file in label_files:
        disease_name = os.path.basename(file).replace('_labels_.csv', '')
        try:
            df = pd.read_csv(file, dtype={'eid': str})
            eids = set(df['eid'].tolist())
            disease_eids[disease_name] = eids & set(expression_data_index)
            logging.info(f"Number of samples for disease {disease_name}: {len(disease_eids[disease_name])}")
        except Exception as e:
            logging.error(f"Error reading file {file}: {e}")
    return disease_eids

def get_all_eids_and_labels(disease_eids, healthy_eids, expression_data_index):
    """Get all participant eids and their corresponding labels."""
    all_eids = set.union(*disease_eids.values(), healthy_eids) & set(expression_data_index)
    eid_labels = {}
    for eid in all_eids:
        found = False
        for disease, eids in disease_eids.items():
            if eid in eids:
                eid_labels[eid] = disease
                found = True
                break
        if not found and eid in healthy_eids:
            eid_labels[eid] = 'healthy'
    return list(eid_labels.keys()), list(eid_labels.values())

def perform_tsne(features, labels, save_path, disease_name):
    """Perform t-SNE dimensionality reduction on CLS embeddings and plot a scatter plot."""
    logging.info("Starting t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    tsne_results = tsne.fit_transform(features)
    logging.info("t-SNE dimensionality reduction completed.")

    # Create DataFrame for plotting
    df_tsne = pd.DataFrame({
        'TSNE1': tsne_results[:, 0],
        'TSNE2': tsne_results[:, 1],
        'Label': labels
    })

    # Define color mapping
    unique_labels = sorted(list(set(labels)))
    if len(unique_labels) == 2:
        palette = ['lightblue', 'lightcoral']  # Light blue and light red
    else:
        palette = sns.color_palette("hsv", len(unique_labels))

    # Plot t-SNE scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='TSNE1', y='TSNE2',
        hue='Label',
        palette=palette,
        data=df_tsne,
        legend="full",
        alpha=0.7,
        edgecolor='none'  # Remove white transparent border
    )
    plt.title(f'CLS embeddings for {disease_name}')
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.legend(title='Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved t-SNE scatter plot to {save_path}")

    # Calculate and record clustering metrics
    if len(set(labels)) > 1:
        try:
            silhouette_avg = silhouette_score(tsne_results, labels)
            davies_bouldin = davies_bouldin_score(tsne_results, labels)
            logging.info(f"Silhouette Score: {silhouette_avg:.4f}")
            logging.info(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
            metrics_path = save_path.replace('.pdf', '_tsne_metrics.txt')
            with open(metrics_path, 'w') as f:
                f.write(f"Silhouette Score: {silhouette_avg:.4f}\n")
                f.write(f"Davies-Bouldin Index: {davies_bouldin:.4f}\n")
            logging.info(f"Saved clustering metrics to {metrics_path}")
        except Exception as e:
            logging.error(f"Error calculating clustering metrics: {e}")
    else:
        logging.warning("Only one unique label present, unable to compute clustering metrics.")

if __name__ == "__main__":
    logging.info("Starting to load data...")

    # Load expression data
    try:
        expression_data = pd.read_csv('/path/proteomic_data.csv', dtype={'eid': str})
        expression_data.set_index('eid', inplace=True)
        logging.info(f"Expression data loaded, number of samples: {expression_data.shape[0]}, number of proteins: {expression_data.shape[1]}")
    except Exception as e:
        logging.error(f"Error loading expression data: {e}")
        exit(1)

    # Load protein sequence features
    try:
        with np.load('//path/global_representations.npz') as data:
            protein_features = {key: data[key] for key in data.files}
        logging.info(f"Protein sequence features loaded, total proteins: {len(protein_features)}")
    except Exception as e:
        logging.error(f"Error loading protein sequence features: {e}")
        exit(1)

    try:
        protein_correlation_matrix = pd.read_csv('/path/protein_correlation_matrix.csv', index_col=0)
        logging.info(f"Bias matrix loaded, shape: {protein_correlation_matrix.shape}")
    except Exception as e:
        logging.error(f"Error loading bias matrix: {e}")
        exit(1)

    # Filter common proteins
    common_proteins = sorted(
        list(
            set(expression_data.columns) &
            set(protein_features.keys()) &
            set(protein_correlation_matrix.index)
        )
    )
    logging.info(f"Number of common proteins: {len(common_proteins)}")

    expression_data = expression_data[common_proteins]
    logging.info(f"Number of proteins in expression data (after filtering): {len(expression_data.columns)}")

    protein_features_filtered = {protein: protein_features[protein] for protein in common_proteins}
    protein_correlation_matrix = protein_correlation_matrix.loc[common_proteins, common_proteins]
    logging.info(f"Shape of bias matrix (after filtering): {protein_correlation_matrix.shape}")

    # Convert protein features to tensor
    try:
        features = np.stack([protein_features_filtered[protein] for protein in common_proteins])
        features_tensor = torch.tensor(features, dtype=torch.float32)
        logging.info(f"Converted protein features to tensor, shape: {features_tensor.shape}")
    except Exception as e:
        logging.error(f"Error converting protein features to tensor: {e}")
        exit(1)

    # Convert bias matrix to tensor and pad
    try:
        bias_matrix = torch.tensor(protein_correlation_matrix.values, dtype=torch.float32)
        logging.info(f"Converted bias matrix to tensor, shape: {bias_matrix.shape}")
        bias_matrix = F.pad(bias_matrix, (1, 0, 1, 0), "constant", 0)
        logging.info(f"Shape of bias matrix (with [CLS] token): {bias_matrix.shape}")
    except Exception as e:
        logging.error(f"Error converting bias matrix to tensor: {e}")
        exit(1)

    # Read all disease label files
    label_dir = '/path/labels'
    label_files = [os.path.join(label_dir, f)
                   for f in os.listdir(label_dir)
                   if f.endswith('_labels.csv')]
    if not label_files:
        logging.error(f"No label files found in directory {label_dir}.")
        exit(1)
    logging.info(f"Found {len(label_files)} disease label files.")

    disease_eids_dict = extract_disease_labels(label_files, expression_data.index)

    # Load healthy control eids
    try:
        healthy_df = pd.read_csv('/path/healthy_control.csv', dtype={'eid': str})
        healthy_eids = set(healthy_df['eid'].tolist()) & set(expression_data.index)
        logging.info(f"Healthy controls loaded, number of healthy samples: {len(healthy_eids)}")
    except Exception as e:
        logging.error(f"Error loading healthy control eids: {e}")
        exit(1)

    # Get list of protein names
    protein_names = common_proteins

    # Perform clustering visualization for each disease
    for disease_name, disease_eids in disease_eids_dict.items():
        logging.info(f"Starting processing disease: {disease_name}")

        # Load prediction result file
        preds_csv_path = f'predicted_probabilities_{disease_name}.csv'
        if not os.path.exists(preds_csv_path):
            logging.error(f"Prediction result file {preds_csv_path} does not exist, skipping disease {disease_name}")
            continue

        try:
            df_preds = pd.read_csv(preds_csv_path, dtype={'eid': str})
            logging.info(f"Loaded prediction results, number of samples: {df_preds.shape[0]}")
        except Exception as e:
            logging.error(f"Error loading prediction file {preds_csv_path}: {e}")
            continue

        # Get test eids and labels
        test_eids = df_preds['eid'].tolist()
        test_labels = [1 if label == disease_name else 0 for label in df_preds['True Label']]
        logging.info(f"Number of test samples: {len(test_eids)}")

        # Create dataset and data loader
        test_dataset = ProteinClassificationDataset(expression_data, test_eids, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)

        # Initialize model
        hidden_size = 768
        num_layers = 24
        num_proteins = features_tensor.size(0)
        feature_dim = features_tensor.size(1)
        chunk_size = 512

        base_model = ProteinBERTModel(
            feature_dim=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_proteins=num_proteins,
            features_tensor=features_tensor,
            bias_matrix=bias_matrix,
            chunk_size=chunk_size
        )
        classifier_head = ClassifierHead(hidden_size, 2).to(device)

        # Load saved model weights
        best_base_model_path = f'best_base_model_{disease_name}.pt'
        best_classifier_head_path = f'best_classifier_head_{disease_name}.pt'

        if not os.path.exists(best_base_model_path) or not os.path.exists(best_classifier_head_path):
            logging.error(f"Model file {best_base_model_path} or {best_classifier_head_path} does not exist, skipping disease {disease_name}")
            continue

        try:
            base_model.load_state_dict(torch.load(best_base_model_path, map_location='cpu'), strict=False)
            logging.info(f"Loaded best base model weights from: {best_base_model_path}")
        except Exception as e:
            logging.error(f"Error loading base model weights {best_base_model_path}: {e}")
            continue

        try:
            classifier_head.load_state_dict(torch.load(best_classifier_head_path, map_location='cpu'), strict=False)
            logging.info(f"Loaded best classifier head weights from: {best_classifier_head_path}")
        except Exception as e:
            logging.error(f"Error loading classifier head weights {best_classifier_head_path}: {e}")
            continue

        base_model.to(device)
        classifier_head.to(device)

        # Extract CLS embeddings
        logging.info(f"Starting extraction of CLS embeddings for {disease_name}")
        base_model.eval()
        classifier_head.eval()
        all_features = []
        all_labels = []
        with torch.no_grad():
            for expressions, y, eids in tqdm(test_loader, desc=f"Extracting CLS embeddings [{disease_name}]", leave=False):
                expressions = expressions.to(device)
                cls_emb = base_model(expressions)  # [B, H]
                all_features.append(cls_emb.cpu().numpy())
                all_labels.extend(y.tolist())
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.array(all_labels)
        logging.info(f"CLS embeddings extraction completed, feature shape: {all_features.shape}")

        # Clustering visualization (t-SNE)
        tsne_save_path = f'tsne_{disease_name}.pdf'
        perform_tsne(all_features, all_labels, tsne_save_path, disease_name)

    logging.info("All diseases processed.")