import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import logging
from transformers import BertConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, roc_curve, auc

# Set logging level and format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Check for available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

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

        # Dynamically align bias_matrix_chunk with attention_mask based on chunk size.
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
            bias_matrix_chunk=kwargs.get('bias_matrix_chunk', None),
            bias_coef=kwargs.get('bias_coef', None)
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
        for i in range(total_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, num_proteins)
            chunk_len = end - start

            feature_chunk = feature_embeds[:, start:end, :]
            expr_chunk = expression_embeds[:, start:end, :]

            # Dynamically create attention_mask_chunk based on actual chunk size
            attention_mask_chunk = torch.zeros((batch_size, chunk_len), dtype=torch.long, device=expressions.device)
            attention_mask_chunk[:, :chunk_len] = 1

            rows = [0] + list(range(1 + start, 1 + end))
            # bias_matrix_chunk shape should be [chunk_len+1, chunk_len+1]
            bias_matrix_chunk = self.bias_matrix_full[rows][:, rows]

            cls_emb, attentions = self.forward_one_chunk_batch(feature_chunk, expr_chunk, bias_matrix_chunk,
                                                               attention_mask_chunk,
                                                               output_attentions=output_attentions)
            cls_list.append(cls_emb)
            if output_attentions:
                # Store attentions along with the corresponding sequence length (including CLS token)
                attention_list.append((attentions, chunk_len + 1))
            seq_lengths.append(chunk_len + 1)  # +1 for CLS token

        cls_all = torch.stack(cls_list, dim=0).mean(dim=0)  # [B, H]
        if output_attentions:
            # Handle varying sequence lengths by zero-padding attentions to the maximum length.
            max_seq_len = max(seq_lengths)
            num_layers = len(attention_list[0][0])
            batch_size = expressions.size(0)
            merged_attentions = []
            for layer_idx in range(num_layers):
                padded_attns = torch.zeros((batch_size, 8, max_seq_len, max_seq_len), device=expressions.device)
                for attn_chunk, seq_len in attention_list:
                    attn = attn_chunk[layer_idx]  # [B, Heads, Seq_len, Seq_len]
                    pad_size = max_seq_len - attn.size(-1)
                    attn = F.pad(attn, (0, pad_size, 0, pad_size))
                    padded_attns += attn
                merged_attentions.append(padded_attns / len(attention_list))
            return cls_all, merged_attentions
        else:
            return cls_all

class ClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

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

def predict_new_sample(new_sample_file, protein_names, features_tensor, bias_matrix, model_dir='.', device='cpu'):
    """
    Predict disease probabilities for a new sample given its proteome file.
    """
    try:
        new_sample_data = pd.read_csv(new_sample_file)
    except Exception as e:
        logging.error(f"Error reading new sample file: {e}")
        return


    logging.info("Starting prediction for new sample.")
    expr_values = new_sample_data.iloc[0].values.astype(np.float32)  # Proteome values
    expr_tensor = torch.tensor(expr_values).unsqueeze(0)  # [1, num_proteins]
    expr_tensor = expr_tensor.to(device)

    disease_probabilities = {}

    # Get all disease names from model files (or from filename inference)
    disease_names = [fname.replace('_labels.csv', '') for fname in os.listdir(model_dir) if fname.endswith('_labels.csv')]
    if not disease_names:
        model_files = [f for f in os.listdir(model_dir) if f.startswith('best_base_model_') and f.endswith('.pt')]
        disease_names = [f.replace('best_base_model_', '').replace('.pt', '') for f in model_files]

    for disease_name in tqdm(disease_names, desc="Predicting disease probabilities"):
        base_model_path = os.path.join(model_dir, f'best_base_model_{disease_name}.pt')
        classifier_head_path = os.path.join(model_dir, f'best_classifier_head_{disease_name}.pt')

        if not os.path.exists(base_model_path) or not os.path.exists(classifier_head_path):
            logging.warning(f"Missing model file: {base_model_path} or {classifier_head_path}. Skipping prediction for {disease_name}.")
            continue

        # Initialize base model
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
        base_model.to(device)
        base_model.eval()
        base_model.load_state_dict(torch.load(base_model_path, map_location=device), strict=False)

        # Initialize classifier head
        classifier_head = ClassifierHead(hidden_size, 2)
        classifier_head.to(device)
        classifier_head.eval()
        classifier_head.load_state_dict(torch.load(classifier_head_path, map_location=device))

        with torch.no_grad():
            cls_emb = base_model(expr_tensor)  # [1, H]
            logits = classifier_head(cls_emb)  # [1, 2]
            probs = torch.softmax(logits, dim=1)  # [1, 2]
            prob_disease = probs[:, 1].item()  # Probability for disease class
            disease_probabilities[disease_name] = prob_disease

    if disease_probabilities:
        df_inference = pd.DataFrame(list(disease_probabilities.items()), columns=['Disease', 'Probability'])
        df_inference = df_inference.sort_values(by='Probability', ascending=False).reset_index(drop=True)
        print("\nNew sample disease probabilities:")
        print(df_inference)
        df_inference.to_csv('new_sample_predictions.csv', index=False)
        logging.info("Saved new sample predictions to new_sample_predictions.csv")
    else:
        logging.warning("No disease predictions available.")

if __name__ == "__main__":
    logging.info("Loading data...")

    # Load expression data
    expression_data = pd.read_csv('/path/proteomic_data.csv', dtype={'eid': str})
    expression_data.set_index('eid', inplace=True)

    # Load protein features
    with np.load('/path/global_representations.npz') as data:
        protein_features = {key: data[key] for key in data.files}

    # Load protein correlation matrix
    protein_correlation_matrix = pd.read_csv('/path/protein_correlation_matrix.csv', index_col=0)

    # Determine common proteins
    common_proteins = sorted(
        list(
            set(expression_data.columns) &
            set(protein_features.keys()) &
            set(protein_correlation_matrix.index)
        )
    )
    expression_data = expression_data[common_proteins]
    protein_features_filtered = {p: protein_features[p] for p in common_proteins}
    protein_correlation_matrix = protein_correlation_matrix.loc[common_proteins, common_proteins]

    # Create features_tensor and bias_matrix
    features = np.stack([protein_features_filtered[p] for p in common_proteins])
    features_tensor = torch.tensor(features, dtype=torch.float32)
    bias_matrix = torch.tensor(protein_correlation_matrix.values, dtype=torch.float32)
    # Pad so that subsequent indexing for [chunk_len+1, chunk_len+1] works (no fixed size of 512)
    bias_matrix = F.pad(bias_matrix, (1, 0, 1, 0), "constant", 0)

    # Get list of protein names
    protein_names = ['CLS'] + common_proteins  # Add CLS token

    # New sample proteome file
    new_sample_file = '/path/new_sample_proteomic_data.csv'

    model_directory = '/path/models'

    # Run prediction for the new sample
    predict_new_sample(new_sample_file, protein_names, features_tensor, bias_matrix, model_dir=model_directory, device=device)