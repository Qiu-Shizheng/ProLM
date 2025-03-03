import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertConfig, get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna
import warnings
from torch.nn.parallel import DataParallel
import logging
import seaborn as sns
import random
import csv
import math
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertModel, BertEncoder, BertLayer, BertSelfAttention

# Ignore warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.info(f"Using device: {device}, GPU count: {gpu_count}")
else:
    device = torch.device("cpu")
    gpu_count = 0
    logging.info("Using CPU for training")

logging.info("Loading healthy subjects' EID list...")
healthy_eids = pd.read_csv('/path/healthy_eids.csv')['eid'].tolist()

logging.info("Loading proteomic expression data for all participants...")
expression_data = pd.read_csv('/path/proteomic.csv')

logging.info("Filtering data for healthy subjects...")
expression_data = expression_data[expression_data['eid'].isin(healthy_eids)]
expression_data.set_index('eid', inplace=True)

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
logging.info(f"Shape of bias matrix (after filtering): {protein_correlation_matrix.shape}")

logging.info("Converting protein features to tensor...")
features = np.stack([protein_features_filtered[protein] for protein in common_proteins])
features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
logging.info(f"Shape of feature tensor: {features_tensor.shape}")

bias_matrix = torch.tensor(protein_correlation_matrix.values, dtype=torch.float32, device=device)
logging.info(f"Shape of bias matrix (without [CLS] token): {bias_matrix.shape}")
bias_matrix = F.pad(bias_matrix, (1, 0, 1, 0), "constant", 0)
logging.info(f"Shape of bias matrix (with [CLS] token): {bias_matrix.shape}")

# Range for random values for masking (80% masked with 0, 10% randomly replaced, rest unchanged)
random_min = expression_data.min().min()
random_max = expression_data.max().max()

class ProteinExpressionDataset(Dataset):
    def __init__(self, expression_data):
        self.expression_data = expression_data.values
        self.num_samples = expression_data.shape[0]
        self.num_proteins = expression_data.shape[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        expressions = self.expression_data[idx]
        return torch.tensor(expressions, dtype=torch.float32)

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
        past_key_value=None,
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

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot pass both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("One of input_ids or inputs_embeds must be provided")

        batch_size, seq_length = input_shape[0], input_shape[1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones((encoder_batch_size, encoder_sequence_length), device=inputs_embeds.device)
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
        self.output_layer = nn.Linear(hidden_size, 1)

        self.register_buffer('features_tensor', features_tensor)
        self.register_buffer('bias_matrix_full', bias_matrix)
        self.bias_coef = nn.Parameter(torch.tensor(1.0))

    def forward_one_chunk(self, feature_embeds_chunk, expr_embeds_chunk, bias_matrix_chunk, attention_mask_chunk):
        batch_size, chunk_len, _ = feature_embeds_chunk.size()
        embeddings = feature_embeds_chunk * expr_embeds_chunk  # Element-wise multiplication fusion (Hadamard product)

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

        prediction_scores_chunk = self.output_layer(sequence_output[:, 1:, :]).squeeze(-1)
        return prediction_scores_chunk, attentions

    def forward(self, expressions, attention_mask):
        batch_size = expressions.size(0)
        num_proteins = self.num_proteins

        feature_embeds = self.feature_embedding(self.features_tensor)
        feature_embeds = feature_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        expressions = expressions.unsqueeze(-1)
        expression_embeds = self.expression_embedding(expressions)

        if num_proteins <= self.chunk_size:
            bias_matrix_chunk = self.bias_matrix_full
            prediction_scores, attentions = self.forward_one_chunk(feature_embeds, expression_embeds, bias_matrix_chunk, attention_mask)
            return prediction_scores, attentions
        else:
            prediction_chunks = []
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

                prediction_scores_chunk, attentions_chunk = self.forward_one_chunk(
                    feature_embeds_chunk,
                    expr_embeds_chunk,
                    bias_matrix_chunk,
                    attention_mask_chunk
                )
                prediction_chunks.append(prediction_scores_chunk)
                if attentions_chunk is not None:
                    all_attentions.append(attentions_chunk)

            prediction_scores = torch.cat(prediction_chunks, dim=1)
            attentions = all_attentions[-1] if all_attentions else None
            return prediction_scores, attentions

def train_model(trial):
    batch_size = trial.suggest_categorical('batch_size', [16, 24])
    hidden_size = trial.suggest_categorical('hidden_size', [512, 768])
    num_layers = trial.suggest_categorical('num_layers', [12, 24])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-4)
    num_epochs = 5

    dataset = ProteinExpressionDataset(expression_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    feature_dim = features_tensor.size(1)
    num_proteins = features_tensor.size(0)

    chunk_size = 512
    model = ProteinBERTModel(feature_dim, hidden_size, num_layers, num_proteins, features_tensor, bias_matrix, chunk_size=chunk_size)

    if gpu_count > 1:
        model = DataParallel(model)

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Adding learning rate scheduler: Warmup + Cosine Decay
    total_steps = len(dataloader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    mask_stats = {
        'total_masked': 0,
        'mask_as_zero': 0,
        'mask_as_random': 0,
        'mask_kept': 0
    }

    tolerance = 0.3375
    total_correct = 0
    total_predicted = 0

    sum_of_squared_errors = 0.0
    sum_of_abs_errors = 0.0
    sum_of_actual = 0.0
    sum_of_actual_sqr = 0.0

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Trial {trial.number} Epoch {epoch + 1}/{num_epochs}")
        for expressions in progress_bar:
            expressions = expressions.to(device)
            attention_mask = torch.ones(expressions.size(), dtype=torch.long, device=device)

            probability_matrix = torch.full(expressions.size(), 0.15, device=device)
            masked_indices = torch.bernoulli(probability_matrix).bool()

            replace_prob = torch.rand(masked_indices.sum(), device=device)
            mask_as_zero = replace_prob < 0.8
            mask_as_random = (replace_prob >= 0.8) & (replace_prob < 0.9)

            num_masked = masked_indices.sum().item()
            if num_masked == 0:
                continue

            num_mask_as_zero = mask_as_zero.sum().item()
            num_mask_as_random = mask_as_random.sum().item()
            num_mask_kept = num_masked - num_mask_as_zero - num_mask_as_random

            mask_stats['total_masked'] += num_masked
            mask_stats['mask_as_zero'] += num_mask_as_zero
            mask_stats['mask_as_random'] += num_mask_as_random
            mask_stats['mask_kept'] += num_mask_kept

            labels = expressions.clone()
            expressions_masked = expressions.clone()
            expressions_masked[masked_indices] = 0.0

            if mask_as_random.sum() > 0:
                random_values = torch.empty(mask_as_random.sum(), device=device).uniform_(random_min, random_max)
                expressions_masked[masked_indices][mask_as_random] = random_values

            optimizer.zero_grad()
            outputs, _ = model(expressions_masked, attention_mask)
            loss = criterion(outputs[masked_indices], labels[masked_indices])

            predicted_values = outputs[masked_indices]
            actual_values = labels[masked_indices]

            correct = (torch.abs(predicted_values - actual_values) < tolerance).sum().item()
            total_correct += correct
            total_predicted += num_masked

            errors = predicted_values - actual_values
            sum_of_squared_errors += (errors**2).sum().item()
            sum_of_abs_errors += errors.abs().sum().item()

            sum_of_actual += actual_values.sum().item()
            sum_of_actual_sqr += (actual_values**2).sum().item()

            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
        trial.report(avg_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    if total_predicted > 0:
        accuracy = total_correct / total_predicted
    else:
        accuracy = 0.0
    logging.info(f"Mask Prediction Accuracy (tolerance={tolerance}): {accuracy:.4f}")

    total_masked = mask_stats['total_masked']
    if total_masked > 0:
        mse = sum_of_squared_errors / total_masked
        mae = sum_of_abs_errors / total_masked
        sst = sum_of_actual_sqr - (sum_of_actual**2 / total_masked)
        if sst > 0:
            r2 = 1 - (sum_of_squared_errors / sst)
        else:
            r2 = 0.0
        logging.info(f"Final Metrics on Masked Predictions: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    else:
        logging.info("No masked positions found, cannot compute MSE/MAE/R².")

    return avg_loss

logging.info("Starting hyperparameter optimization...")
study = optuna.create_study(direction='minimize')
study.optimize(train_model, n_trials=5)

logging.info("Best hyperparameters:")
logging.info(study.best_params)
logging.info(f"Best loss: {study.best_value:.4f}")

# Using best hyperparameters for full pre-training
best_params = study.best_params
batch_size = best_params['batch_size']
hidden_size = best_params['hidden_size']
num_layers = best_params['num_layers']
learning_rate = best_params['learning_rate']
num_epochs = 100

dataset = ProteinExpressionDataset(expression_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

chunk_size = 512
model = ProteinBERTModel(features_tensor.size(1), hidden_size, num_layers, features_tensor.size(0), features_tensor, bias_matrix, chunk_size=chunk_size)

if gpu_count > 1:
    model = DataParallel(model)

model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Adding learning rate scheduler: Warmup + Cosine Decay
total_steps = len(dataloader) * num_epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

# Lists to record metrics for each epoch
loss_values = []
acc_values = []
mse_values = []
mae_values = []
r2_values = []

model.train()
for epoch in range(num_epochs):
    tolerance = 0.3375
    epoch_loss = 0.0
    epoch_total_correct = 0
    epoch_total_pred = 0
    epoch_sum_squared_errors = 0.0
    epoch_sum_abs_errors = 0.0
    epoch_sum_actual = 0.0
    epoch_sum_actual_sq = 0.0
    epoch_masked_count = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for expressions in progress_bar:
        expressions = expressions.to(device)
        attention_mask = torch.ones(expressions.size(), dtype=torch.long, device=device)

        probability_matrix = torch.full(expressions.size(), 0.15, device=device)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        if masked_indices.sum().item() == 0:
            continue

        replace_prob = torch.rand(masked_indices.sum(), device=device)
        mask_as_zero = replace_prob < 0.8
        mask_as_random = (replace_prob >= 0.8) & (replace_prob < 0.9)

        labels = expressions.clone()
        expressions_masked = expressions.clone()
        expressions_masked[masked_indices] = 0.0

        if mask_as_random.sum() > 0:
            random_values = torch.empty(mask_as_random.sum(), device=device).uniform_(random_min, random_max)
            expressions_masked[masked_indices][mask_as_random] = random_values

        num_masked = masked_indices.sum().item()
        optimizer.zero_grad()
        outputs, _ = model(expressions_masked, attention_mask)
        loss = criterion(outputs[masked_indices], labels[masked_indices])
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        epoch_loss += loss.item()

        predicted_values = outputs[masked_indices]
        actual_values = labels[masked_indices]

        correct = (torch.abs(predicted_values - actual_values) < tolerance).sum().item()
        epoch_total_correct += correct
        epoch_total_pred += num_masked

        errors = predicted_values - actual_values
        epoch_sum_squared_errors += (errors**2).sum().item()
        epoch_sum_abs_errors += errors.abs().sum().item()
        epoch_sum_actual += actual_values.sum().item()
        epoch_sum_actual_sq += (actual_values**2).sum().item()
        epoch_masked_count += num_masked

        progress_bar.set_postfix({'Batch Loss': loss.item()})

    epoch_avg_loss = epoch_loss / len(dataloader)
    loss_values.append(epoch_avg_loss)

    if epoch_total_pred > 0:
        epoch_acc = epoch_total_correct / epoch_total_pred
    else:
        epoch_acc = 0.0
    acc_values.append(epoch_acc)

    if epoch_masked_count > 0:
        mse_epoch = epoch_sum_squared_errors / epoch_masked_count
        mae_epoch = epoch_sum_abs_errors / epoch_masked_count
        sst = epoch_sum_actual_sq - (epoch_sum_actual**2 / epoch_masked_count)
        if sst > 0:
            r2_epoch = 1 - (epoch_sum_squared_errors / sst)
        else:
            r2_epoch = 0.0
    else:
        mse_epoch, mae_epoch, r2_epoch = 0.0, 0.0, 0.0

    mse_values.append(mse_epoch)
    mae_values.append(mae_epoch)
    r2_values.append(r2_epoch)

    logging.info(f"Epoch {epoch + 1:3d} - Loss: {epoch_avg_loss:.4f}, Mask Acc: {epoch_acc:.4f}, MSE: {mse_epoch:.4f}, MAE: {mae_epoch:.4f}, R²: {r2_epoch:.4f}")

best_epoch_loss = np.argmin(loss_values) + 1
best_loss_value = min(loss_values)
best_epoch_acc = np.argmax(acc_values) + 1
best_acc_value = max(acc_values)
best_epoch_mse = np.argmin(mse_values) + 1
best_mse_value = min(mse_values)
best_epoch_mae = np.argmin(mae_values) + 1
best_mae_value = min(mae_values)
best_epoch_r2 = np.argmax(r2_values) + 1
best_r2_value = max(r2_values)

logging.info("====== Best Metrics ======")
logging.info(f"Best Loss: {best_loss_value:.4f} at epoch {best_epoch_loss}")
logging.info(f"Best Mask Accuracy: {best_acc_value:.4f} at epoch {best_epoch_acc}")
logging.info(f"Best MSE: {best_mse_value:.4f} at epoch {best_epoch_mse}")
logging.info(f"Best MAE: {best_mae_value:.4f} at epoch {best_epoch_mae}")
logging.info(f"Best R²: {best_r2_value:.4f} at epoch {best_epoch_r2}")

# Plot metric curves

# 1. Pre-training Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_values, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Pre-training Loss')
plt.grid(True)
plt.savefig('pretraining_loss.pdf')
plt.show()

# 2. Mask Prediction Accuracy Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), acc_values, marker='o', color='green')
plt.xlabel('Epoch')
plt.ylabel('Mask Prediction Accuracy')
plt.title('Mask Prediction Accuracy vs Epoch')
plt.grid(True)
plt.savefig('mask_accuracy.pdf')
plt.show()

# 3. MSE Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), mse_values, marker='o', color='red')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE vs Epoch')
plt.grid(True)
plt.savefig('mse.pdf')
plt.show()

# 4. MAE Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), mae_values, marker='o', color='orange')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('MAE vs Epoch')
plt.grid(True)
plt.savefig('mae.pdf')
plt.show()

# 5. R² Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), r2_values, marker='o', color='purple')
plt.xlabel('Epoch')
plt.ylabel('R²')
plt.title('R² vs Epoch')
plt.grid(True)
plt.savefig('r2.pdf')
plt.show()

logging.info("Saving pre-training model...")
if gpu_count > 1:
    torch.save(model.module.state_dict(), 'ProLM_pretraining.pt')
else:
    torch.save(model.state_dict(), 'ProLM_pretraining.pt')

torch.save(features_tensor, 'features_tensor_ProLM_pretraining.pt')
torch.save(bias_matrix, 'bias_matrix_ProLM_pretraining.pt')

logging.info("Starting interpretability analysis: Extracting attention weights for all protein features...")

if isinstance(model, DataParallel):
    model = model.module

model.eval()

num_proteins = features_tensor.size(0)
sum_attention_weights = torch.zeros(num_proteins, num_proteins)
num_samples = 0

with torch.no_grad():
    for expressions in tqdm(dataloader, desc="Extracting attention weights"):
        expressions = expressions.to(device)
        attention_mask = torch.ones(expressions.size(), dtype=torch.long, device=device)

        batch_size = expressions.size(0)
        feature_embeds = model.feature_embedding(model.features_tensor)
        feature_embeds_batched = feature_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        expressions_ = expressions.unsqueeze(-1)
        expression_embeds = model.expression_embedding(expressions_)

        total_chunks = (num_proteins + model.chunk_size - 1) // model.chunk_size
        chunk_attentions_list = []
        for i in range(total_chunks):
            start = i * model.chunk_size
            end = min((i + 1) * model.chunk_size, num_proteins)

            feature_embeds_chunk = feature_embeds_batched[:, start:end, :]
            expr_embeds_chunk = expression_embeds[:, start:end, :]
            attention_mask_chunk = attention_mask[:, start:end]

            rows = [0] + list(range(1 + start, 1 + end))
            cols = rows
            bias_matrix_chunk = model.bias_matrix_full[rows][:, cols]

            outputs_chunk, attentions_chunk = model.forward_one_chunk(
                feature_embeds_chunk,
                expr_embeds_chunk,
                bias_matrix_chunk,
                attention_mask_chunk
            )

            avg_attentions = []
            for att in attentions_chunk:
                mean_attention = att.mean(dim=1)
                avg_attentions.append(mean_attention)
            avg_attention = torch.stack(avg_attentions).mean(dim=0)

            chunk_len = end - start
            chunk_attentions_list.append((avg_attention, chunk_len))

        big_matrix = torch.zeros(batch_size, num_proteins, num_proteins, device=device)
        current_offset = 0
        for (chunk_attention, chunk_len) in chunk_attentions_list:
            protein_attention = chunk_attention[:, 1:, 1:]
            big_matrix[:, current_offset:current_offset + chunk_len,
            current_offset:current_offset + chunk_len] = protein_attention
            current_offset += chunk_len

        sum_attention_weights += big_matrix.sum(dim=0).cpu()
        num_samples += batch_size

mean_attention_weights = sum_attention_weights / num_samples
protein_importance = mean_attention_weights.sum(dim=0).numpy()
protein_names = common_proteins

importance_df = pd.DataFrame({
    'Protein': protein_names,
    'Attention_Weight': protein_importance
})

importance_df = importance_df.sort_values(by='Attention_Weight', ascending=False)
importance_df.to_csv('protein_importance_ProLM_pretraining.csv', index=False)
logging.info("All protein feature weight results saved to protein_importance_ProLM_pretraining.csv")

logging.info("Top 20 important proteins:")
print(importance_df.head(20))

logging.info("Plotting bar chart for important proteins...")
plt.figure(figsize=(12, 8))
sns.barplot(x='Attention_Weight', y='Protein', data=importance_df.head(20))
plt.title('Top 20 Important Proteins by Attention Weights')
plt.xlabel('Attention Weight')
plt.ylabel('Protein')
plt.tight_layout()
plt.savefig('important_proteins_ProLM_pretraining.pdf')
plt.show()