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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import pickle

# Set logging level and format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
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
        # At this point, seq_len should match bias_matrix_chunk
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
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions=kwargs.get('output_attentions', False),
            bias_matrix_chunk=kwargs.get('bias_matrix_chunk', None),
            bias_coef=kwargs.get('bias_coef', None)
        )
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

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, inputs_embeds.size()[:-1], device=inputs_embeds.device
        )
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
        # Fuse features using element-wise multiplication
        embeddings = feature_embeds_chunk * expr_embeds_chunk
        B, chunk_len, H = embeddings.shape
        cls_tokens = self.cls_token.expand(B, 1, H)
        # Concatenate CLS token
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
        # To track which proteins are in which chunk
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
            # bias_matrix_chunk should match shape [chunk_len+1, chunk_len+1]
            bias_matrix_chunk = self.bias_matrix_full[rows][:, rows]

            cls_emb, attentions = self.forward_one_chunk_batch(
                feature_chunk, expr_chunk, bias_matrix_chunk, attention_mask_chunk,
                output_attentions=output_attentions
            )
            cls_list.append(cls_emb)
            if output_attentions:
                # Store attentions along with their corresponding indices
                attention_list.append((attentions, start, end))
            seq_lengths.append(chunk_len + 1)  # +1 for CLS token

        cls_all = torch.stack(cls_list, dim=0).mean(dim=0)  # [B, H]
        if output_attentions:
            # Collect all attentions from all chunks
            aggregate_attention = torch.zeros(num_proteins, device=expressions.device)
            total_attention = torch.zeros(num_proteins, device=expressions.device)

            for attentions, start, end in attention_list:
                cls_attentions = []
                for layer_attn in attentions:
                    # layer_attn: [batch_size, num_heads, seq_len, seq_len]
                    cls_attn = layer_attn[0, :, 0, 1:]
                    cls_attn_mean = cls_attn.mean(dim=0)  # [seq_len]
                    cls_attentions.append(cls_attn_mean)
                cls_attentions = torch.stack(cls_attentions, dim=0).mean(dim=0)  # [seq_len]
                aggregate_attention[start:end] = cls_attentions
                total_attention[start:end] += 1

            total_attention = torch.clamp(total_attention, min=1)
            aggregate_attention = aggregate_attention / total_attention

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
        return torch.tensor(expr), label, eid  # return eid


class ClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def extract_disease_labels(label_files, expression_data_index):
    disease_eids = {}
    for file in label_files:
        disease_name = os.path.basename(file).replace('_labels.csv', '')
        df = pd.read_csv(file, dtype={'eid': str})
        eids = set(df['eid'].tolist())
        disease_eids[disease_name] = eids & set(expression_data_index)
    return disease_eids


def get_all_eids_and_labels(disease_eids, healthy_eids, expression_data_index):
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


def evaluate(base_model, classifier_head, loader, device, label2id, output_attentions=False, dataset_type='Test'):
    base_model.eval()
    classifier_head.eval()
    preds = []
    trues = []
    features_list = []
    probs_list = []
    attentions_list = []
    eids_list = []
    with torch.no_grad():
        for expressions, y, eids in tqdm(loader, desc=f"Evaluating on [{dataset_type} set]", leave=False):
            expressions = expressions.to(device)
            y = y.to(device)
            if output_attentions:
                cls_emb, attentions = base_model(expressions, output_attentions=True)
                attentions_list.append(attentions)
            else:
                cls_emb = base_model(expressions)
            logits = classifier_head(cls_emb)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=-1)
            preds.extend(pred.cpu().tolist())
            trues.extend(y.cpu().tolist())
            features_list.append(cls_emb.cpu().numpy())
            probs_list.append(probs.cpu().numpy())
            eids_list.extend(eids)
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='binary', zero_division=0)
    precision = precision_score(trues, preds, average='binary', zero_division=0)
    if len(set(trues)) == 2:
        y_scores = np.concatenate(probs_list, axis=0)[:, 1]
        auc_score = roc_auc_score(trues, y_scores)
    else:
        auc_score = float('nan')
        y_scores = []
    features_array = np.concatenate(features_list, axis=0)
    base_model.train()
    classifier_head.train()
    results = {
        'acc': acc,
        'f1': f1,
        'precision': precision,
        'auc': auc_score,
        'trues': trues,
        'preds': preds,
        'probs': y_scores if len(set(trues)) == 2 else [],
        'features': features_array,
        'eids': eids_list
    }
    if output_attentions:
        results['attentions'] = attentions_list
    return results


def train_classifier_epoch(base_model, classifier_head, loader, optimizer, criterion, epoch, device, losses,
                           scaler=None):
    base_model.train()
    classifier_head.train()
    total_loss = 0.0
    preds = []
    trues = []
    probs_list = []
    eids_list = []
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch + 1}", leave=False)
    for batch_idx, (expressions, y, eids) in progress_bar:
        expressions = expressions.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                cls_emb = base_model(expressions)
                logits = classifier_head(cls_emb)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            cls_emb = base_model(expressions)
            logits = classifier_head(cls_emb)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        losses.append(loss.item())

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=-1)
            preds.extend(pred.cpu().tolist())
            trues.extend(y.cpu().tolist())
            probs_list.append(probs.cpu().numpy())
            eids_list.extend(eids)
    if len(probs_list) > 0 and len(trues) > 0:
        if len(set(trues)) == 2:
            y_scores = np.concatenate(probs_list, axis=0)[:, 1]
        else:
            y_scores = []
    else:
        y_scores = []
    train_results = {
        'trues': trues,
        'preds': preds,
        'probs': y_scores,
        'eids': eids_list
    }
    return total_loss / len(loader), train_results


def visualize_attention(aggregate_attention, protein_names, save_heatmap_path, save_barplot_path, top_k=10):

    attention_df = pd.DataFrame({
        'Protein': protein_names,
        'Attention': aggregate_attention
    })

    attention_csv_path = save_barplot_path.replace('.pdf', '_attention_weights_bert.csv')
    attention_df.to_csv(attention_csv_path, index=False)
    logging.info(f"Saved attention weights to {attention_csv_path}")

    top_attention = attention_df.sort_values(by='Attention', ascending=False).head(top_k)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Attention', y='Protein', data=top_attention, palette='viridis')
    plt.title(f'Top {top_k} Proteins by Attention Weight')
    plt.xlabel('Average Attention Weight')
    plt.ylabel('Protein')
    plt.tight_layout()
    plt.savefig(save_barplot_path)
    plt.close()
    logging.info(f"Saved top {top_k} attention bar plot to {save_barplot_path}")


if __name__ == "__main__":
    logging.info("Loading data...")
    # Load expression data
    expression_data = pd.read_csv('/path/proteomic_data.csv', dtype={'eid': str})
    expression_data.set_index('eid', inplace=True)

    with np.load('/path/global_representations.npz') as data:
        protein_features = {key: data[key] for key in data.files}
    protein_correlation_matrix = pd.read_csv('/path/protein_correlation_matrix.csv', index_col=0)

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

    features = np.stack([protein_features_filtered[p] for p in common_proteins])
    features_tensor = torch.tensor(features, dtype=torch.float32)
    bias_matrix = torch.tensor(protein_correlation_matrix.values, dtype=torch.float32)
    bias_matrix = F.pad(bias_matrix, (1, 0, 1, 0), "constant", 0)

    # Read all disease label files
    label_files = [os.path.join('/path/labels', f)
                   for f in os.listdir('/path/labels')
                   if f.endswith('_labels.csv')]

    disease_eids_dict = extract_disease_labels(label_files, expression_data.index)

    # Read healthy control eids
    healthy_df = pd.read_csv('/path/healthy_control.csv', dtype={'eid': str})
    healthy_eids = set(healthy_df['eid'].tolist()) & set(expression_data.index)

    protein_names = common_proteins

    # Define a global list to store best metrics for all disease models
    all_disease_metrics = []

    # Process each disease for training and prediction
    for disease_name, disease_eids in disease_eids_dict.items():
        logging.info(f"Processing disease: {disease_name}")

        current_eids = list(disease_eids | healthy_eids)
        eid_labels = []
        for eid in current_eids:
            if eid in disease_eids:
                eid_labels.append(1)  # Disease labeled as 1
            else:
                eid_labels.append(0)  # Healthy labeled as 0

        if len(current_eids) == 0:
            logging.warning(f"Disease {disease_name} has no samples, skipping")
            continue

        # Split into training and test sets
        train_ratio = 0.8
        total_count = len(current_eids)
        train_count = int(total_count * train_ratio)
        indices = list(range(total_count))
        random.shuffle(indices)
        train_indices = indices[:train_count]
        test_indices = indices[train_count:]

        train_eids = [current_eids[i] for i in train_indices]
        train_labels = [eid_labels[i] for i in train_indices]
        test_eids = [current_eids[i] for i in test_indices]
        test_labels = [eid_labels[i] for i in test_indices]

        logging.info(f"Training set size: {len(train_eids)}, Test set size: {len(test_eids)}")

        # Create datasets and data loaders
        train_dataset = ProteinClassificationDataset(expression_data, train_eids, train_labels)
        test_dataset = ProteinClassificationDataset(expression_data, test_eids, test_labels)

        batch_size = 96
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

        # Initialize models
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
        base_model.load_state_dict(torch.load('/path/ProLM_pretraining.pt', map_location='cpu'), strict=False)
        base_model.to(device)

        if torch.cuda.device_count() > 1:
            base_model = nn.DataParallel(base_model)

        # Freeze model parameters; if using DataParallel, access the original model via .module
        if isinstance(base_model, nn.DataParallel):
            model_for_freeze = base_model.module
        else:
            model_for_freeze = base_model
        for param in model_for_freeze.parameters():
            param.requires_grad = False

        # Unfreeze the last two layers of the BERT encoder
        for layer in model_for_freeze.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        num_classes = 2  # Binary classification
        classifier_head = ClassifierHead(hidden_size, num_classes).to(device)
        if torch.cuda.device_count() > 1:
            classifier_head = nn.DataParallel(classifier_head)

        for param in classifier_head.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(
            list(classifier_head.parameters()) + list(filter(lambda p: p.requires_grad, model_for_freeze.parameters())),
            lr=1e-4
        )
        criterion = nn.CrossEntropyLoss()

        best_auc = 0.0
        best_test_results = None
        best_train_results = None
        epochs = 10
        losses = []
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(epochs):
            logging.info(f"Training Epoch {epoch + 1}/{epochs}")
            train_loss, train_results = train_classifier_epoch(
                base_model, classifier_head, train_loader, optimizer, criterion, epoch,
                device, losses, scaler=scaler
            )
            test_results = evaluate(base_model, classifier_head, test_loader, device, {0: 0, 1: 1}, dataset_type='Test')
            acc = test_results['acc']
            f1 = test_results['f1']
            auc_score = test_results['auc']
            precision = test_results['precision']
            if not np.isnan(auc_score) and auc_score > best_auc:
                best_auc = auc_score
                if isinstance(base_model, nn.DataParallel):
                    torch.save(base_model.module.state_dict(), f'best_base_model_{disease_name}.pt')
                    torch.save(classifier_head.module.state_dict(), f'best_classifier_head_{disease_name}.pt')
                else:
                    torch.save(base_model.state_dict(), f'best_base_model_{disease_name}.pt')
                    torch.save(classifier_head.state_dict(), f'best_classifier_head_{disease_name}.pt')
                best_test_results = test_results
                best_train_results = train_results
            logging.info(
                f"Epoch {epoch + 1}/{epochs}: Train Loss={train_loss:.4f}, Test Acc={acc:.4f}, Precision={precision:.4f}, F1={f1:.4f}, AUC={auc_score:.4f}"
            )

        if best_test_results is None:
            logging.warning(f"AUC could not be computed for {disease_name} in all epochs, possibly due to a single class in the test set.")
            continue

        logging.info(f"{disease_name} best test AUC={best_auc:.4f}")

        # Record best test metrics for the current disease into the global list
        all_disease_metrics.append({
            "Disease": disease_name,
            "Test_Accuracy": best_test_results['acc'],
            "Test_F1": best_test_results['f1'],
            "Test_Precision": best_test_results['precision'],
            "Test_AUC": best_test_results['auc']
        })

        # Merge predictions from training and test sets
        all_eids = best_train_results['eids'] + best_test_results['eids']
        all_trues = best_train_results['trues'] + best_test_results['trues']
        all_preds = best_train_results['preds'] + best_test_results['preds']
        if len(best_train_results['probs']) > 0 and len(best_test_results['probs']) > 0:
            all_probs = np.concatenate([best_train_results['probs'], best_test_results['probs']])
        else:
            all_probs = []

        df_preds = pd.DataFrame({
            'eid': all_eids,
            'True Label': ['Healthy' if label == 0 else disease_name for label in all_trues],
            'Predicted Label': ['Healthy' if pred == 0 else disease_name for pred in all_preds],
            'Probability': all_probs
        })
        df_preds.to_csv(f'predicted_probabilities_{disease_name}.csv', index=False)
        logging.info(f"Saved predictions for {disease_name} to predicted_probabilities_{disease_name}.csv")

        # ROC curve plotting
        if len(set(test_results['trues'])) == 2:
            y_true = test_results['trues']
            y_scores = test_results['probs']
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([-0.05, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC for {disease_name}')
            plt.legend(loc="lower right")
            plt.savefig(f'bert_roc_curve_{disease_name}.pdf')
            plt.close()
            logging.info(f"Saved ROC curve for {disease_name} to bert_roc_curve_{disease_name}.pdf")
        else:
            logging.warning(f"Test set for {disease_name} has only one class; cannot plot ROC curve.")

        # Attention weights visualization
        logging.info(f"Starting attention visualization for {disease_name}")
        visualize_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
        aggregate_attention = np.zeros(len(protein_names), dtype=np.float32)
        attention_count = np.zeros(len(protein_names), dtype=np.float32)

        with torch.no_grad():
            for expressions_vis, y_vis, eids_vis in tqdm(visualize_loader, desc=f"Visualizing attention weights for {disease_name}", leave=False):
                expressions_vis = expressions_vis.to(device)
                # If the model uses DataParallel, call the underlying module's forward method
                if isinstance(base_model, nn.DataParallel):
                    cls_emb_vis, attentions_vis = base_model.module.forward(expressions_vis, output_attentions=True)
                else:
                    cls_emb_vis, attentions_vis = base_model(expressions_vis, output_attentions=True)

                logits_vis = classifier_head(cls_emb_vis)
                probs_vis = torch.softmax(logits_vis, dim=1)
                preds_vis = logits_vis.argmax(dim=-1).cpu().tolist()

                # At this point, attentions_vis should have shape matching the number of proteins
                aggregate_attention += attentions_vis.cpu().numpy()
                attention_count += 1

        attention_count = np.maximum(attention_count, 1)
        aggregate_attention /= attention_count

        heatmap_path = f'attention_heatmap_{disease_name}.pdf'
        barplot_path = f'attention_barplot_{disease_name}_top10.pdf'
        visualize_attention(aggregate_attention, protein_names, heatmap_path, barplot_path, top_k=10)

    # Summarize all disease model metrics into a CSV file
    if all_disease_metrics:
        summary_df = pd.DataFrame(all_disease_metrics)
        summary_df.to_csv("disease_model_metrics_summary.csv", index=False)
        logging.info("Saved all disease model metrics to disease_model_metrics_summary.csv")

    logging.info("All diseases processed.")