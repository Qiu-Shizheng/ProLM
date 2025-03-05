import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import BertConfig, get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import seaborn as sns
import random
import math
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.manifold import TSNE
import umap
from transformers.models.bert.modeling_bert import BertModel, BertEncoder, BertLayer, BertSelfAttention


# Set up logging
logging.basicConfig(
    filename='fine_tuning.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_count = torch.cuda.device_count()
    logging.info(f"Using device: {device}, Number of GPUs: {gpu_count}")
else:
    device = torch.device("cpu")
    gpu_count = 0
    logging.info("Using CPU for training")

logging.info("Loading protein expression data for all participants...")
expression_data = pd.read_csv('/path/proteomic_data.csv')

logging.info(f"Number of proteins in expression data (before filtering): {len(expression_data.columns)}")

logging.info("Loading protein sequence feature representations...")
with np.load('/path/global_representations.npz') as data:
    protein_features = {key: data[key] for key in data.files}

logging.info(f"Number of protein features: {len(protein_features)}")

logging.info("Loading bias matrix...")
protein_correlation_matrix = pd.read_csv('/path/protein_correlation_matrix.csv', index_col=0)
logging.info(f"Number of proteins in bias matrix: {protein_correlation_matrix.shape[0]}")

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
logging.info(f"Shape of bias matrix after filtering: {protein_correlation_matrix.shape}")

logging.info("Converting protein features to tensor...")
features = np.stack([protein_features_filtered[protein] for protein in common_proteins])
features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
logging.info(f"Shape of feature tensor: {features_tensor.shape}")

bias_matrix = torch.tensor(protein_correlation_matrix.values, dtype=torch.float32, device=device)
logging.info(f"Shape of bias matrix (without [CLS] token): {bias_matrix.shape}")
bias_matrix = F.pad(bias_matrix, (1, 0, 1, 0), "constant", 0)
logging.info(f"Shape of bias matrix (with [CLS] token): {bias_matrix.shape}")

class DiabetesDataset(Dataset):
    def __init__(self, expression_data, labels, eids):
        self.expression_data = expression_data
        self.labels = labels
        self.eids = eids  # Store eids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        expressions = self.expression_data[idx]
        label = self.labels[idx]
        eid = self.eids[idx]
        return expressions, label, eid

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

        batch_size, num_heads, seq_len, _ = attention_scores.size()
        if bias_matrix_chunk is not None and bias_coef is not None:
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

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)

        return outputs

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
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        bias_matrix_chunk=None,
        bias_coef=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            bias_matrix_chunk=bias_matrix_chunk,
            bias_coef=bias_coef
        )

        attention_output = self.output(self_outputs[0])
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)

        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class CustomBertLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = CustomBertAttention(config)
        if self.is_decoder:
            self.crossattention = CustomBertAttention(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        bias_matrix_chunk=None,
        bias_coef=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            bias_matrix_chunk=bias_matrix_chunk,
            bias_coef=bias_coef
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
                bias_matrix_chunk=bias_matrix_chunk,
                bias_coef=bias_coef
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        return outputs

class CustomBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([CustomBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        bias_matrix_chunk=None,
        bias_coef=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values[i] if past_key_values is not None else None,
                output_attentions=output_attentions,
                bias_matrix_chunk=bias_matrix_chunk,
                bias_coef=bias_coef
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_attentions] if v is not None)
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': next_decoder_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
        }

class CustomBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = CustomBertEncoder(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        bias_matrix_chunk=None,
        bias_coef=None,
    ):
        # 修改自BertModel
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("cannot pass input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("must pass in one of input_ids or inputs_embeds")

        batch_size, seq_length = input_shape[0], input_shape[1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.float32, device=inputs_embeds.device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones((encoder_batch_size, encoder_sequence_length), dtype=torch.float32, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = inputs_embeds
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            bias_matrix_chunk=bias_matrix_chunk,
            bias_coef=bias_coef,
        )

        sequence_output = encoder_outputs['last_hidden_state']
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return {
            'last_hidden_state': sequence_output,
            'pooler_output': pooled_output,
            'past_key_values': encoder_outputs['past_key_values'],
            'hidden_states': encoder_outputs['hidden_states'],
            'attentions': encoder_outputs['attentions'],
        }

class ProteinBERTModel(nn.Module):
    def __init__(self, feature_dim, hidden_size, num_layers, num_proteins, features_tensor, bias_matrix, chunk_size=512):
        super(ProteinBERTModel, self).__init__()
        self.chunk_size = chunk_size
        self.num_proteins = num_proteins
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

        self.config = config

    def forward_one_chunk(self, feature_embeds_chunk, expr_embeds_chunk, bias_matrix_chunk, attention_mask_chunk):
        batch_size, chunk_len, _ = feature_embeds_chunk.size()
        embeddings = feature_embeds_chunk * expr_embeds_chunk

        cls_tokens = self.cls_token.expand(batch_size, 1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        attention_mask_chunk = torch.cat((torch.ones(batch_size, 1, device=attention_mask_chunk.device), attention_mask_chunk), dim=1)

        outputs = self.bert(
            inputs_embeds=embeddings,
            attention_mask=attention_mask_chunk,
            output_attentions=True,
            bias_matrix_chunk=bias_matrix_chunk,
            bias_coef=self.bias_coef
        )

        sequence_output = outputs['last_hidden_state']
        attentions = outputs['attentions']

        cls_output = sequence_output[:, 0, :]  # (batch_size, hidden_size)
        return cls_output, attentions

    def forward(self, expressions, attention_mask=None):
        batch_size = expressions.size(0)
        num_proteins = self.num_proteins

        feature_embeds = self.feature_embedding(self.features_tensor)
        feature_embeds = feature_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        expressions = expressions.unsqueeze(-1)
        expression_embeds = self.expression_embedding(expressions)

        if attention_mask is None:
            attention_mask = torch.ones(expressions.size(), dtype=torch.float32, device=expressions.device)

        if num_proteins <= self.chunk_size:
            bias_matrix_chunk = self.bias_matrix_full
            cls_output, attentions = self.forward_one_chunk(feature_embeds, expression_embeds, bias_matrix_chunk, attention_mask)
            return cls_output, attentions
        else:
            cls_outputs = []
            all_attentions = []
            total_chunks = (num_proteins + self.chunk_size - 1) // self.chunk_size

            for i in range(total_chunks):
                start = i * self.chunk_size
                end = min((i + 1) * self.chunk_size, num_proteins)

                feature_embeds_chunk = feature_embeds[:, start:end, :]
                expr_embeds_chunk = expression_embeds[:, start:end, :]
                attention_mask_chunk = attention_mask[:, start:end]

                rows = [0] + list(range(1 + start, 1 + end))
                cols = rows
                bias_matrix_chunk = self.bias_matrix_full[rows][:, cols]

                cls_output_chunk, attentions_chunk = self.forward_one_chunk(
                    feature_embeds_chunk,
                    expr_embeds_chunk,
                    bias_matrix_chunk,
                    attention_mask_chunk
                )
                cls_outputs.append(cls_output_chunk)
                if attentions_chunk is not None:
                    all_attentions.append(attentions_chunk)

            cls_output = torch.stack(cls_outputs, dim=1).mean(dim=1)  # (batch_size, hidden_size)
            attentions = all_attentions[-1] if all_attentions else None
            return cls_output, attentions


class ProteinBERTForClassification(nn.Module):
    def __init__(self, pre_trained_model, num_labels=2):
        super(ProteinBERTForClassification, self).__init__()
        self.bert = pre_trained_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(pre_trained_model.config.hidden_size, num_labels)

    def forward(self, expressions, attention_mask=None):

        cls_output, attentions = self.bert(expressions, attention_mask=attention_mask)
        # Use [CLS] token's output for classification
        cls_output = self.dropout(cls_output)  # (batch_size, hidden_size)
        logits = self.classifier(cls_output)  # (batch_size, num_labels)
        return logits, attentions, cls_output

def main():
    logging.info("Starting fine-tuning of pre-trained model...")

    # Load pre-trained model related data
    logging.info("Loading pre-trained model related data...")
    features_tensor = torch.load('/path/features_tensor_ProLM_pretraining.pt').to(device)
    bias_matrix = torch.load('/path/bias_matrix_ProLM_pretraining.pt').to(device)

    # Load pre-trained model
    logging.info("Loading pre-trained ProteinBERTModel...")
    hidden_size = 768  # Based on pre-training
    num_layers = 24    # Based on pre-training
    num_proteins = features_tensor.size(0)
    chunk_size = 512

    # Create BERT config
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

    # Initialize pre-trained model
    pre_trained_model = ProteinBERTModel(
        feature_dim=features_tensor.size(1),
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_proteins=num_proteins,
        features_tensor=features_tensor,
        bias_matrix=bias_matrix,
        chunk_size=chunk_size
    )

    # Load pre-trained weights, ignoring mismatched keys
    model_path = '/path/ProLM_pretraining.pt'
    if os.path.exists(model_path):
        pre_trained_state_dict = torch.load(model_path, map_location=device)

        pre_trained_state_dict = {k: v for k, v in pre_trained_state_dict.items() if not k.startswith('output_layer')}
        pre_trained_model.load_state_dict(pre_trained_state_dict, strict=False)
        logging.info(f"Successfully loaded pre-trained model weights from {model_path}, ignoring unmatched keys.")
    else:
        logging.error(f"Pre-trained model weight file {model_path} does not exist!")
        return

    # Create classification model
    model = ProteinBERTForClassification(pre_trained_model, num_labels=2)

    if gpu_count > 1:
        model = nn.DataParallel(model)
        logging.info(f"Using DataParallel for model parallelism, Number of GPUs: {gpu_count}")

    model.to(device)

    # Load label data
    logging.info("Loading label data...")
    type1_df = pd.read_csv('/path/Type_1_Diabetes_participants.csv')
    type2_df = pd.read_csv('/path/Type_2_Diabetes_participants.csv')

    type1_eids = set(type1_df['eid'].tolist())
    type2_eids = set(type2_df['eid'].tolist())

    duplicate_eids = type1_eids.intersection(type2_eids)
    if duplicate_eids:
        logging.info(f"Found {len(duplicate_eids)} duplicate eids in Type 1 and Type 2 Diabetes labels. Removing these participants.")
        type1_eids = type1_eids - duplicate_eids
        type2_eids = type2_eids - duplicate_eids
    else:
        logging.info("No duplicate eids found in Type 1 and Type 2 Diabetes labels.")

    logging.info(f"Number of Type 1 Diabetes samples after removing duplicates: {len(type1_eids)}")
    logging.info(f"Number of Type 2 Diabetes samples after removing duplicates: {len(type2_eids)}")

    # Load expression data and set index to eid
    logging.info("Loading expression data and setting index to eid...")
    expression_data_full = pd.read_csv('/path/proteomic_data.csv')
    expression_data_full = expression_data_full.set_index('eid')

    logging.info("Ensuring expression data has the common proteins...")
    common_proteins = sorted(
        list(
            set(expression_data_full.columns) &
            set(protein_features.keys()) &
            set(protein_correlation_matrix.index)
        )
    )
    logging.info(f"Number of common proteins: {len(common_proteins)}")

    expression_data_full = expression_data_full[common_proteins]
    logging.info(f"Number of proteins in expression data after filtering: {len(expression_data_full.columns)}")

    # Filter participants with labels
    logging.info("Filtering participants with labels...")
    type1_data = expression_data_full.loc[expression_data_full.index.intersection(type1_eids)]
    type2_data = expression_data_full.loc[expression_data_full.index.intersection(type2_eids)]

    # Combine data and create labels
    logging.info("Creating labels...")
    type1_labels = [0] * len(type1_data)
    type2_labels = [1] * len(type2_data)

    combined_data = pd.concat([type1_data, type2_data], axis=0)
    combined_labels = type1_labels + type2_labels

    logging.info(f"Total number of samples: {len(combined_labels)}")
    logging.info(f"Number of Type 1 Diabetes samples: {len(type1_labels)}")
    logging.info(f"Number of Type 2 Diabetes samples: {len(type2_labels)}")

    combined_eids = combined_data.index.tolist()

    # Convert to tensors
    expressions_tensor = torch.tensor(combined_data.values, dtype=torch.float32)
    labels_tensor = torch.tensor(combined_labels, dtype=torch.long)

    # Create Dataset
    dataset = DiabetesDataset(expressions_tensor, labels_tensor, combined_eids)

    # Split into training and validation sets
    validation_split = 0.2
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}")

    # Define DataLoader
    batch_size = 24
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

    num_epochs = 10
    total_steps = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # Unfreeze the last two layers of the model
    logging.info("Unfreezing the last two layers of the model...")
    if isinstance(model, nn.DataParallel):
        target_model = model.module.bert.bert
    else:
        target_model = model.bert.bert

    for name, param in target_model.named_parameters():
        param.requires_grad = False

    for name, param in target_model.encoder.layer[-2:].named_parameters():
        param.requires_grad = True
        logging.info(f"Unfrozen parameter: {name}")

    for name, param in target_model.named_parameters():
        if not param.requires_grad:
            logging.debug(f"Frozen parameter: {name}")

    # Training and validation functions
    def train_epoch(model, dataloader, optimizer, criterion, scheduler, device):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for expressions, labels, _ in tqdm(dataloader, desc="Training", leave=False):
            expressions = expressions.to(device)
            labels = labels.to(device)

            # Create attention_mask, all ones indicating all positions are valid
            attention_mask = torch.ones(expressions.size(), dtype=torch.float32, device=device)

            optimizer.zero_grad()
            logits, _, _ = model(expressions, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        return avg_loss, accuracy

    def eval_epoch(model, dataloader, criterion, device):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        all_probs = []
        all_attentions = []
        all_cls_outputs = []
        all_eids = []

        with torch.no_grad():
            for expressions, labels, eids in tqdm(dataloader, desc="Validation", leave=False):
                expressions = expressions.to(device)
                labels = labels.to(device)

                # Create attention_mask, all ones indicating all positions are valid
                attention_mask = torch.ones(expressions.size(), dtype=torch.float32, device=device)

                logits, attentions, cls_output = model(expressions, attention_mask=attention_mask)
                loss = criterion(logits, labels)

                total_loss += loss.item()

                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy()[:, 1])  # Probability for Type 2 Diabetes

                if attentions is not None:

                    cls_attentions = []
                    for layer_att in attentions:

                        cls_to_protein_att = layer_att[:, :, 0, 1:].cpu().numpy()  # [batch_size, num_heads, num_proteins]
                        cls_attentions.append(cls_to_protein_att)

                    cls_attentions = np.stack(cls_attentions, axis=0)  # [num_layers, batch_size, num_heads, num_proteins]

                    cls_attentions = cls_attentions.mean(axis=2)  # [num_layers, batch_size, num_proteins]

                    cls_attentions = cls_attentions.mean(axis=0)  # [batch_size, num_proteins]
                    all_attentions.extend(cls_attentions)
                else:
                    # If no attentions, append zeros
                    all_attentions.extend([np.zeros(num_proteins) for _ in range(expressions.size(0))])

                # Collect cls_output
                all_cls_outputs.extend(cls_output.cpu().numpy())

                # Collect eids
                all_eids.extend(eids.tolist())

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        # Calculate other metrics
        precision = precision_score(all_labels, all_preds, average='binary', pos_label=1)
        recall = recall_score(all_labels, all_preds, average='binary', pos_label=1)
        f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)
        auc = roc_auc_score(all_labels, all_probs)

        return avg_loss, accuracy, precision, recall, f1, auc, all_probs, all_labels, all_preds, all_eids, all_attentions, all_cls_outputs

    best_auc = 0.0
    best_model_path = 'fine_tuned_model.pt'
    all_epoch_metrics = []

    # Initialize variables to store the best epoch's metrics
    best_val_metrics = None
    best_val_probs = None
    best_val_labels = None
    best_val_preds = None
    best_val_eids = None
    best_val_attentions = None
    best_val_cls_outputs = None

    logging.info("Starting training...")
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        logging.info(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, val_probs, val_labels, val_preds, val_eids, val_attentions, val_cls_outputs = eval_epoch(model, val_loader, criterion, device)
        logging.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        all_epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'val_auc': val_auc
        })

        if val_auc > best_auc:
            best_auc = val_auc
            best_val_metrics = {
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'val_auc': val_auc
            }
            best_val_probs = val_probs
            best_val_labels = val_labels
            best_val_preds = val_preds
            best_val_eids = val_eids
            best_val_attentions = val_attentions
            best_val_cls_outputs = val_cls_outputs

            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Saved the best model (AUC: {best_auc:.4f}) to {best_model_path}")

    logging.info("Training completed!")

    # Plot training and validation loss curves
    epochs = [m['epoch'] for m in all_epoch_metrics]
    train_losses = [m['train_loss'] for m in all_epoch_metrics]
    val_losses = [m['val_loss'] for m in all_epoch_metrics]

    # Plot validation AUC curve
    val_aucs = [m['val_auc'] for m in all_epoch_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_aucs, label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('validation_auc.pdf')
    plt.show()

    # Plot ROC curve for the best epoch
    if best_val_metrics is not None:
        fpr, tpr, thresholds = roc_curve(best_val_labels, best_val_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {best_auc:.4f})")
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve of the Best Model')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig('best_model_ROC_curve.pdf')
        plt.show()

    # Save predictions with true labels and predicted labels
    if best_val_eids is not None:
        logging.info("Saving predictions with true labels and predicted labels...")
        predictions_df = pd.DataFrame({
            'eid': best_val_eids,
            'true_label': best_val_labels,
            'pred_label': best_val_preds,
            'prob_type2_diabetes': best_val_probs
        })
        predictions_df.to_csv('diabetes_predictions_with_labels.csv', index=False)
        logging.info("Predictions with labels have been saved to diabetes_predictions_with_labels.csv")

    # Extract and save attention weights, and visualize top 20 protein features
    if best_val_attentions is not None:
        logging.info("Processing and saving attention weights...")
        # best_val_attentions: list of attention scores for each sample [num_samples, num_proteins]
        # Convert to numpy array
        attention_array = np.array(best_val_attentions)  # Shape: [num_samples, num_proteins]

        expected_attentions = len(common_proteins)
        actual_attentions = attention_array.shape[1]
        if actual_attentions != expected_attentions:
            logging.error(f"Mismatch in attentions: expected {expected_attentions}, got {actual_attentions}")
            # Handle the mismatch by padding or truncating
            if actual_attentions < expected_attentions:
                padding = np.zeros((attention_array.shape[0], expected_attentions - actual_attentions))
                attention_array = np.hstack((attention_array, padding))
                logging.info(f"Padded attentions to match expected {expected_attentions} proteins.")
            else:
                attention_array = attention_array[:, :expected_attentions]
                logging.info(f"Truncated attentions to match expected {expected_attentions} proteins.")

        # Save all CLS attentions with eids
        logging.info("Saving all CLS attention weights...")
        attention_df = pd.DataFrame(attention_array, columns=common_proteins)
        attention_df.insert(0, 'eid', best_val_eids)
        attention_df.to_csv('all_cls_attentions.csv', index=False)
        logging.info("All CLS attention weights have been saved to all_cls_attentions.csv")

        # Compute average attention per protein
        avg_attention = attention_array.mean(axis=0)  # Shape: [num_proteins]

        # Get top 20 proteins
        top20_indices = avg_attention.argsort()[-20:][::-1]
        top20_attention = avg_attention[top20_indices]
        top20_proteins = [common_proteins[i] for i in top20_indices]

        # Create a DataFrame for top 20 attention
        top20_df = pd.DataFrame({
            'protein': top20_proteins,
            'average_attention': top20_attention
        })
        top20_df.to_csv('top20_proteins_attention.csv', index=False)
        logging.info("Top 20 proteins' average attention have been saved to top20_proteins_attention.csv")

        # Plot the top 20 attention as a bar plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='average_attention', y='protein', data=top20_df, palette='viridis')
        plt.xlabel('Average Attention')
        plt.ylabel('Protein')
        plt.title('Top 20 Proteins by Average Attention')
        plt.tight_layout()
        plt.savefig('top20_proteins_attention_barplot.pdf')
        plt.show()

    # Add t-SNE and UMAP dimensionality reduction cluster plots
    if best_val_cls_outputs is not None:
        logging.info("Performing t-SNE and UMAP dimensionality reduction...")
        cls_embeddings = np.array(best_val_cls_outputs)  # [num_samples, hidden_size]

        # t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        cls_tsne = tsne.fit_transform(cls_embeddings)
        logging.info("t-SNE dimensionality reduction completed.")

        # Get labels
        labels = np.array(best_val_labels)

        # t-SNE cluster plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=cls_tsne[:,0],
            y=cls_tsne[:,1],
            hue=labels,
            palette=['blue', 'orange'],
            alpha=0.7
        )
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('t-SNE Dimensionality Reduction Cluster Plot')
        plt.legend(title='Type 2 Diabetes', loc='best')
        plt.grid(True)
        plt.savefig('tsne_cluster_plot.pdf')
        plt.show()

    # Save training metrics
    logging.info("Saving training metrics for all epochs...")
    metrics_df = pd.DataFrame(all_epoch_metrics)
    metrics_df.to_csv('training_metrics.csv', index=False)
    logging.info("Training metrics have been saved to training_metrics.csv")

    # Final evaluation metrics
    if best_val_metrics is not None:
        logging.info("Final Evaluation Metrics:")
        logging.info(f"Loss: {best_val_metrics['val_loss']:.4f}")
        logging.info(f"Accuracy: {best_val_metrics['val_accuracy']:.4f}")
        logging.info(f"Precision: {best_val_metrics['val_precision']:.4f}")
        logging.info(f"Recall: {best_val_metrics['val_recall']:.4f}")
        logging.info(f"F1 Score: {best_val_metrics['val_f1']:.4f}")
        logging.info(f"AUC: {best_val_metrics['val_auc']:.4f}")

if __name__ == "__main__":
    main()
