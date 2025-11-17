#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import BertConfig, get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from torch.nn.parallel import DataParallel
import logging
import seaborn as sns
import random
import csv
import math

from transformers.models.bert.modeling_bert import BertModel, BertEncoder, BertLayer, BertSelfAttention

warnings.filterwarnings("ignore")

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

results_dir = "results_ProLM"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
logging.info(f"All output files will be saved in: {results_dir}")

logging.info("Loading healthy participant EID list...")
healthy_eids = pd.read_csv('/your/path/healthy_eids.csv')['eid'].tolist()

logging.info("Loading protein expression data for all participants...")
expression_data = pd.read_csv('/your/path/proteomic_data.csv')

logging.info("Filtering data to healthy participants...")
expression_data = expression_data[expression_data['eid'].isin(healthy_eids)]
expression_data.set_index('eid', inplace=True)

logging.info(f"Number of proteins before filtering: {len(expression_data.columns)}")

logging.info("Loading protein sequence representations...")
with np.load('/your/path/global_representations.npz') as data:
    protein_features = {key: data[key] for key in data.files}

logging.info(f"Number of protein feature vectors: {len(protein_features)}")

logging.info("Loading correlation bias matrix...")
protein_correlation_matrix = pd.read_csv('/your/path/protein_correlation_matrix.csv', index_col=0)
logging.info(f"Number of proteins in bias matrix: {protein_correlation_matrix.shape[0]}")

common_proteins = sorted(
    list(
        set(expression_data.columns) &
        set(protein_features.keys()) &
        set(protein_correlation_matrix.index)
    )
)
logging.info(f"Number of shared proteins: {len(common_proteins)}")

expression_data = expression_data[common_proteins]
logging.info(f"Number of proteins after filtering: {len(expression_data.columns)}")

protein_features_filtered = {protein: protein_features[protein] for protein in common_proteins}
protein_correlation_matrix = protein_correlation_matrix.loc[common_proteins, common_proteins]
logging.info(f"Filtered bias matrix shape: {protein_correlation_matrix.shape}")

logging.info("Converting protein features to tensor...")
features = np.stack([protein_features_filtered[protein] for protein in common_proteins])
features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
logging.info(f"Feature tensor shape: {features_tensor.shape}")

bias_matrix = torch.tensor(protein_correlation_matrix.values, dtype=torch.float32, device=device)
logging.info(f"Bias matrix shape without [CLS]: {bias_matrix.shape}")
bias_matrix = F.pad(bias_matrix, (1, 0, 1, 0), "constant", 0)
logging.info(f"Bias matrix shape with [CLS]: {bias_matrix.shape}")

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
            return tuple(
                v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_attentions] if v is not None)
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
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        batch_size, seq_length = input_shape[0], input_shape[1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones((encoder_batch_size, encoder_sequence_length),
                                                    device=inputs_embeds.device)
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
    def __init__(self, feature_dim, hidden_size, num_layers, num_proteins, features_tensor, bias_matrix,
                 chunk_size=512):
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
        embeddings = feature_embeds_chunk * expr_embeds_chunk

        cls_tokens = self.cls_token.expand(batch_size, 1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        attention_mask_chunk = torch.cat(
            (torch.ones(batch_size, 1, device=attention_mask_chunk.device), attention_mask_chunk), dim=1)

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
            prediction_scores, attentions = self.forward_one_chunk(feature_embeds, expression_embeds, bias_matrix_chunk,
                                                                   attention_mask)
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


batch_size = 25
hidden_size = 768
num_layers = 24
learning_rate = 2e-5
num_epochs = 200
early_stop_patience = 10
tolerance = 0.5

dataset = ProteinExpressionDataset(expression_data)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=25, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=25, pin_memory=True)
logging.info(f"Total samples: {len(dataset)}, training samples: {train_size}, validation samples: {val_size}")

feature_dim = features_tensor.size(1)
num_proteins = features_tensor.size(0)
chunk_size = 512
model = ProteinBERTModel(feature_dim, hidden_size, num_layers, num_proteins, features_tensor, bias_matrix,
                         chunk_size=chunk_size)
if gpu_count > 1:
    model = DataParallel(model)
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
total_steps = len(train_dataloader) * num_epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                            num_training_steps=total_steps)

train_loss_values = []
train_acc_values = []
train_mse_values = []
train_mae_values = []
train_r2_values = []

val_loss_values = []
val_acc_values = []
val_mse_values = []
val_mae_values = []
val_r2_values = []

best_val_loss = float('inf')
best_epoch = -1
early_stop_counter = 0

logging.info("Starting pre-training...")
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    epoch_train_total_correct = 0
    epoch_train_total_masked = 0
    epoch_train_sum_squared_errors = 0.0
    epoch_train_sum_abs_errors = 0.0
    epoch_train_sum_actual = 0.0
    epoch_train_sum_actual_sq = 0.0

    train_progress = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]")
    for expressions in train_progress:
        expressions = expressions.to(device)
        attention_mask = torch.ones(expressions.size(), dtype=torch.long, device=device)

        probability_matrix = torch.full(expressions.size(), 0.15, device=device)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        if masked_indices.sum().item() == 0:
            continue

        labels = expressions.clone()
        expressions_masked = expressions.clone()
        expressions_masked[masked_indices] = 0.0

        replace_prob = torch.rand(masked_indices.sum(), device=device)
        mask_as_random = (replace_prob >= 0.8) & (replace_prob < 0.9)
        if mask_as_random.sum() > 0:
            random_values = torch.empty(mask_as_random.sum(), device=device).uniform_(random_min, random_max)
            expressions_masked[masked_indices][mask_as_random] = random_values

        optimizer.zero_grad()
        outputs, _ = model(expressions_masked, attention_mask)
        loss = criterion(outputs[masked_indices], labels[masked_indices])
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_train_loss += loss.item()

        predicted_values = outputs[masked_indices]
        actual_values = labels[masked_indices]
        correct = (torch.abs(predicted_values - actual_values) < tolerance).sum().item()
        epoch_train_total_correct += correct
        num_masked = masked_indices.sum().item()
        epoch_train_total_masked += num_masked

        errors = predicted_values - actual_values
        epoch_train_sum_squared_errors += (errors ** 2).sum().item()
        epoch_train_sum_abs_errors += errors.abs().sum().item()
        epoch_train_sum_actual += actual_values.sum().item()
        epoch_train_sum_actual_sq += (actual_values ** 2).sum().item()

        train_progress.set_postfix({'Loss': loss.item()})
    train_avg_loss = epoch_train_loss / len(train_dataloader)
    if epoch_train_total_masked > 0:
        train_acc = epoch_train_total_correct / epoch_train_total_masked
        train_mse = epoch_train_sum_squared_errors / epoch_train_total_masked
        train_mae = epoch_train_sum_abs_errors / epoch_train_total_masked
        sst_train = epoch_train_sum_actual_sq - (epoch_train_sum_actual ** 2 / epoch_train_total_masked)
        train_r2 = 1 - (epoch_train_sum_squared_errors / sst_train) if sst_train > 0 else 0.0
    else:
        train_acc, train_mse, train_mae, train_r2 = 0.0, 0.0, 0.0, 0.0

    train_loss_values.append(train_avg_loss)
    train_acc_values.append(train_acc)
    train_mse_values.append(train_mse)
    train_mae_values.append(train_mae)
    train_r2_values.append(train_r2)

    model.eval()
    epoch_val_loss = 0.0
    epoch_val_total_correct = 0
    epoch_val_total_masked = 0
    epoch_val_sum_squared_errors = 0.0
    epoch_val_sum_abs_errors = 0.0
    epoch_val_sum_actual = 0.0
    epoch_val_sum_actual_sq = 0.0

    with torch.no_grad():
        val_progress = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]")
        for expressions in val_progress:
            expressions = expressions.to(device)
            attention_mask = torch.ones(expressions.size(), dtype=torch.long, device=device)

            probability_matrix = torch.full(expressions.size(), 0.15, device=device)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            if masked_indices.sum().item() == 0:
                continue

            labels = expressions.clone()
            expressions_masked = expressions.clone()
            expressions_masked[masked_indices] = 0.0

            replace_prob = torch.rand(masked_indices.sum(), device=device)
            mask_as_random = (replace_prob >= 0.8) & (replace_prob < 0.9)
            if mask_as_random.sum() > 0:
                random_values = torch.empty(mask_as_random.sum(), device=device).uniform_(random_min, random_max)
                expressions_masked[masked_indices][mask_as_random] = random_values

            outputs, _ = model(expressions_masked, attention_mask)
            loss = criterion(outputs[masked_indices], labels[masked_indices])
            epoch_val_loss += loss.item()

            predicted_values = outputs[masked_indices]
            actual_values = labels[masked_indices]
            correct = (torch.abs(predicted_values - actual_values) < tolerance).sum().item()
            epoch_val_total_correct += correct
            num_masked = masked_indices.sum().item()
            epoch_val_total_masked += num_masked

            errors = predicted_values - actual_values
            epoch_val_sum_squared_errors += (errors ** 2).sum().item()
            epoch_val_sum_abs_errors += errors.abs().sum().item()
            epoch_val_sum_actual += actual_values.sum().item()
            epoch_val_sum_actual_sq += (actual_values ** 2).sum().item()

    val_avg_loss = epoch_val_loss / len(val_dataloader)
    if epoch_val_total_masked > 0:
        val_acc = epoch_val_total_correct / epoch_val_total_masked
        val_mse = epoch_val_sum_squared_errors / epoch_val_total_masked
        val_mae = epoch_val_sum_abs_errors / epoch_val_total_masked
        sst_val = epoch_val_sum_actual_sq - (epoch_val_sum_actual ** 2 / epoch_val_total_masked)
        val_r2 = 1 - (epoch_val_sum_squared_errors / sst_val) if sst_val > 0 else 0.0
    else:
        val_acc, val_mse, val_mae, val_r2 = 0.0, 0.0, 0.0, 0.0

    val_loss_values.append(val_avg_loss)
    val_acc_values.append(val_acc)
    val_mse_values.append(val_mse)
    val_mae_values.append(val_mae)
    val_r2_values.append(val_r2)

    logging.info(f"Epoch {epoch + 1:3d} - Train Loss: {train_avg_loss:.4f}, Train Acc: {train_acc:.4f}, "
                 f"Train MSE: {train_mse:.4f}, Train MAE: {train_mae:.4f}, Train R^2: {train_r2:.4f} || "
                 f"Val Loss: {val_avg_loss:.4f}, Val Acc: {val_acc:.4f}, Val MSE: {val_mse:.4f}, "
                 f"Val MAE: {val_mae:.4f}, Val R^2: {val_r2:.4f}")

    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        best_epoch = epoch + 1
        early_stop_counter = 0
        best_model_state = model.module.state_dict() if gpu_count > 1 else model.state_dict()
        torch.save(best_model_state, os.path.join(results_dir, 'ProLM_model_best.pt'))
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            logging.info(f"Validation loss did not improve for {early_stop_patience} epochs. Stopping early.")
            break

logging.info("===== Best Validation Metrics =====")
logging.info(f"Best Validation Loss: {best_val_loss:.4f} at epoch {best_epoch}")

if gpu_count > 1:
    torch.save(model.module.state_dict(), os.path.join(results_dir, 'ProLM_model_final.pt'))
else:
    torch.save(model.state_dict(), os.path.join(results_dir, 'ProLM_model_final.pt'))

torch.save(features_tensor, os.path.join(results_dir, 'features_tensor_1215.pt'))
torch.save(bias_matrix, os.path.join(results_dir, 'bias_matrix_ProLM.pt'))

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_loss_values) + 1), train_loss_values, marker='o', label='Train Loss')
plt.plot(range(1, len(val_loss_values) + 1), val_loss_values, marker='o', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Pre-training Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'pretraining_loss.pdf'))
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_acc_values) + 1), train_acc_values, marker='o', label='Train Accuracy', color='blue')
plt.plot(range(1, len(val_acc_values) + 1), val_acc_values, marker='o', label='Validation Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Mask Prediction Accuracy')
plt.title('Mask Prediction Accuracy vs Epoch')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'mask_accuracy.pdf'))
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_mse_values) + 1), train_mse_values, marker='o', label='Train MSE', color='blue')
plt.plot(range(1, len(val_mse_values) + 1), val_mse_values, marker='o', label='Validation MSE', color='red')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE vs Epoch')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'mse.pdf'))
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_mae_values) + 1), train_mae_values, marker='o', label='Train MAE', color='blue')
plt.plot(range(1, len(val_mae_values) + 1), val_mae_values, marker='o', label='Validation MAE', color='orange')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('MAE vs Epoch')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'mae.pdf'))
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_r2_values) + 1), train_r2_values, marker='o', label='Train R^2', color='blue')
plt.plot(range(1, len(val_r2_values) + 1), val_r2_values, marker='o', label='Validation R^2', color='purple')
plt.xlabel('Epoch')
plt.ylabel('R^2')
plt.title('R^2 vs Epoch')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'r2.pdf'))
plt.show()

logging.info("Starting interpretability analysis by extracting attention weights...")
all_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

if isinstance(model, DataParallel):
    model = model.module

model.eval()

sum_attention_weights = torch.zeros(num_proteins, num_proteins)
num_samples = 0

with torch.no_grad():
    for expressions in tqdm(all_dataloader, desc="Extracting attention weights"):
        expressions = expressions.to(device)
        attention_mask = torch.ones(expressions.size(), dtype=torch.long, device=device)

        batch_size_curr = expressions.size(0)
        feature_embeds = model.feature_embedding(model.features_tensor)
        feature_embeds_batched = feature_embeds.unsqueeze(0).expand(batch_size_curr, -1, -1)
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

        big_matrix = torch.zeros(batch_size_curr, num_proteins, num_proteins, device=device)
        current_offset = 0
        for (chunk_attention, chunk_len) in chunk_attentions_list:
            protein_attention = chunk_attention[:, 1:, 1:]
            big_matrix[:, current_offset:current_offset + chunk_len,
            current_offset:current_offset + chunk_len] = protein_attention
            current_offset += chunk_len

        sum_attention_weights += big_matrix.sum(dim=0).cpu()
        num_samples += batch_size_curr

mean_attention_weights = sum_attention_weights / num_samples
protein_importance = mean_attention_weights.sum(dim=0).numpy()
protein_names = common_proteins

importance_df = pd.DataFrame({
    'Protein': protein_names,
    'Attention_Weight': protein_importance
})

importance_df = importance_df.sort_values(by='Attention_Weight', ascending=False)
importance_df.to_csv(os.path.join(results_dir, 'protein_importance.csv'), index=False)
logging.info("Saved protein importance scores to protein_importance.csv")
logging.info("Top 10 proteins by attention weight:")
print(importance_df.head(10))

logging.info("Plotting top protein importance chart...")
plt.figure(figsize=(12, 8))
sns.barplot(x='Attention_Weight', y='Protein', data=importance_df.head(10))
plt.title('Top 10 Important Proteins by Attention Weights')
plt.xlabel('Attention Weight')
plt.ylabel('Protein')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'important_proteins.pdf'))
plt.show()