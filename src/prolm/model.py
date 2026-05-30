"""Core ProLM model components.

The backbone combines per-protein expression values with external protein
sequence embeddings and a protein-protein correlation bias matrix.  The class
keeps state-dict names compatible with the original pretraining checkpoint.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertLayer,
    BertModel,
    BertSelfAttention,
)


def _choose_num_heads(hidden_size: int, preferred: int = 8) -> int:
    for heads in range(min(preferred, hidden_size), 0, -1):
        if hidden_size % heads == 0:
            return heads
    return 1


class CorrelationBiasedSelfAttention(BertSelfAttention):
    """BERT self-attention with an additive protein-correlation bias."""

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(  # type: ignore[override]
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
        past_key_value = (key_layer, value_layer) if self.is_decoder else None

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

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

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class CorrelationBiasedAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.self = CorrelationBiasedSelfAttention(config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.pruned_heads: set[int] = set()
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
            bias_coef=bias_coef,
        )
        attention_output = self.output(self_outputs[0])
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)
        return (attention_output,) + self_outputs[1:]


class CorrelationBiasedBertLayer(BertLayer):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.attention = CorrelationBiasedAttention(config)

    def forward(  # type: ignore[override]
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
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            bias_matrix_chunk=bias_matrix_chunk,
            bias_coef=bias_coef,
        )
        attention_output = self_attention_outputs[0]
        layer_outputs = self.feed_forward_chunk(attention_output)
        outputs = (layer_outputs,) + self_attention_outputs[1:]
        return outputs


class CorrelationBiasedBertEncoder(BertEncoder):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.layer = nn.ModuleList([CorrelationBiasedBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(  # type: ignore[override]
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        bias_matrix_chunk=None,
        bias_coef=None,
        **kwargs,
    ):
        all_attentions = [] if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                output_attentions=output_attentions,
                bias_matrix_chunk=bias_matrix_chunk,
                bias_coef=bias_coef,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions.append(layer_outputs[1])

        outputs = {"last_hidden_state": hidden_states}
        if output_attentions:
            outputs["attentions"] = all_attentions
        return outputs


class CorrelationBiasedBertModel(BertModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.encoder = CorrelationBiasedBertEncoder(config)
        self.post_init()

    def forward(  # type: ignore[override]
        self,
        inputs_embeds=None,
        attention_mask=None,
        bias_matrix_chunk=None,
        bias_coef=None,
        output_attentions=False,
    ):
        if inputs_embeds is None:
            raise ValueError("inputs_embeds is required")

        input_shape = inputs_embeds.size()[:-1]
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=inputs_embeds.device)
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask,
            input_shape,
            device=inputs_embeds.device,
        )
        return self.encoder(
            hidden_states=inputs_embeds,
            attention_mask=extended_attention_mask,
            bias_matrix_chunk=bias_matrix_chunk,
            bias_coef=bias_coef,
            output_attentions=output_attentions,
        )


@dataclass(frozen=True)
class CheckpointShape:
    feature_dim: int
    hidden_size: int
    num_layers: int
    chunk_size: int


def infer_checkpoint_shape(state_dict: dict[str, torch.Tensor]) -> CheckpointShape:
    """Infer model dimensions from a ProLM checkpoint state dict."""

    feature_weight = state_dict["feature_embedding.weight"]
    hidden_size, feature_dim = feature_weight.shape
    layer_ids = set()
    prefix = "bert.encoder.layer."
    for key in state_dict:
        if key.startswith(prefix):
            rest = key[len(prefix) :]
            layer_ids.add(int(rest.split(".", 1)[0]))
    if not layer_ids:
        raise ValueError("No BERT encoder layers found in checkpoint.")
    chunk_size = int(state_dict["bert.embeddings.word_embeddings.weight"].shape[0])
    return CheckpointShape(
        feature_dim=int(feature_dim),
        hidden_size=int(hidden_size),
        num_layers=max(layer_ids) + 1,
        chunk_size=chunk_size,
    )


class ProLMBackbone(nn.Module):
    """Transformer backbone for expression plus protein-sequence embeddings."""

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int,
        num_layers: int,
        num_proteins: int,
        features_tensor: torch.Tensor,
        bias_matrix: torch.Tensor,
        chunk_size: int = 512,
        num_attention_heads: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if features_tensor.shape != (num_proteins, feature_dim):
            raise ValueError(
                "features_tensor must have shape "
                f"({num_proteins}, {feature_dim}), got {tuple(features_tensor.shape)}"
            )
        expected_bias = (num_proteins + 1, num_proteins + 1)
        if bias_matrix.shape != expected_bias:
            raise ValueError(f"bias_matrix must have shape {expected_bias}, got {tuple(bias_matrix.shape)}")

        self.num_proteins = num_proteins
        self.chunk_size = chunk_size
        attention_heads = num_attention_heads or _choose_num_heads(hidden_size)
        config = BertConfig(
            vocab_size=chunk_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=attention_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=chunk_size + 1,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            is_decoder=False,
        )
        self.bert = CorrelationBiasedBertModel(config)
        self.feature_embedding = nn.Linear(feature_dim, hidden_size)
        self.expression_embedding = nn.Linear(1, hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.register_buffer("features_tensor", features_tensor.float())
        self.register_buffer("bias_matrix_full", bias_matrix.float())
        self.bias_coef = nn.Parameter(torch.tensor(1.0))

    @property
    def hidden_size(self) -> int:
        return int(self.cls_token.shape[-1])

    def forward_one_chunk_batch(
        self,
        feature_embeds_chunk: torch.Tensor,
        expr_embeds_chunk: torch.Tensor,
        bias_matrix_chunk: torch.Tensor,
        attention_mask_chunk: torch.Tensor,
        output_attentions: bool = False,
    ):
        embeddings = feature_embeds_chunk * expr_embeds_chunk
        batch_size, _, hidden_size = embeddings.shape
        cls_tokens = self.cls_token.expand(batch_size, 1, hidden_size)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        attention_mask_chunk = torch.cat(
            (torch.ones(batch_size, 1, device=attention_mask_chunk.device), attention_mask_chunk),
            dim=1,
        )

        outputs = self.bert(
            inputs_embeds=embeddings,
            attention_mask=attention_mask_chunk,
            bias_matrix_chunk=bias_matrix_chunk,
            bias_coef=self.bias_coef,
            output_attentions=output_attentions,
        )
        cls_emb = outputs["last_hidden_state"][:, 0, :]
        return cls_emb, outputs.get("attentions")

    def forward(self, expressions: torch.Tensor, output_attentions: bool = False):
        batch_size, num_proteins = expressions.size()
        if num_proteins != self.num_proteins:
            raise ValueError(f"Expected {self.num_proteins} proteins, got {num_proteins}.")

        feature_embeds = self.feature_embedding(self.features_tensor)
        feature_embeds = feature_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        expression_embeds = self.expression_embedding(expressions.float().unsqueeze(-1))

        cls_list = []
        attention_list = []
        total_chunks = (num_proteins + self.chunk_size - 1) // self.chunk_size
        for chunk_id in range(total_chunks):
            start = chunk_id * self.chunk_size
            end = min((chunk_id + 1) * self.chunk_size, num_proteins)
            chunk_len = end - start
            attention_mask_chunk = torch.ones((batch_size, chunk_len), dtype=torch.long, device=expressions.device)
            rows = [0] + list(range(1 + start, 1 + end))
            bias_matrix_chunk = self.bias_matrix_full[rows][:, rows]
            cls_emb, attentions = self.forward_one_chunk_batch(
                feature_embeds[:, start:end, :],
                expression_embeds[:, start:end, :],
                bias_matrix_chunk,
                attention_mask_chunk,
                output_attentions=output_attentions,
            )
            cls_list.append(cls_emb)
            if output_attentions:
                attention_list.append((attentions, start, end))

        pooled = torch.stack(cls_list, dim=0).mean(dim=0)
        if not output_attentions:
            return pooled

        aggregate_attention = torch.zeros(num_proteins, device=expressions.device)
        total_attention = torch.zeros(num_proteins, device=expressions.device)
        for attentions, start, end in attention_list:
            if attentions is None:
                continue
            cls_attentions = []
            for layer_attn in attentions:
                cls_attn = layer_attn[:, :, 0, 1:].mean(dim=(0, 1))
                cls_attentions.append(cls_attn)
            aggregate_attention[start:end] = torch.stack(cls_attentions, dim=0).mean(dim=0)
            total_attention[start:end] += 1
        aggregate_attention = aggregate_attention / torch.clamp(total_attention, min=1)
        return pooled, aggregate_attention

    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        cleaned = strip_module_prefix(state_dict)
        return self.load_state_dict(cleaned, strict=strict)


class ProLMClassifier(nn.Module):
    """ProLM backbone with a classification head."""

    def __init__(self, backbone: ProLMBackbone, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone.hidden_size, num_classes),
        )

    def forward(self, expressions: torch.Tensor):
        pooled = self.backbone(expressions)
        return self.classifier(pooled)


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        cleaned[key.removeprefix("module.")] = value
    return cleaned


def load_matching_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    skip_keys: set[str] | None = None,
):
    """Load checkpoint tensors whose names and shapes match the current module."""

    skip_keys = skip_keys or set()
    current = module.state_dict()
    matched = OrderedDict()
    skipped = []
    for key, value in strip_module_prefix(state_dict).items():
        if key in skip_keys:
            skipped.append(key)
        elif key in current and tuple(current[key].shape) == tuple(value.shape):
            matched[key] = value
        else:
            skipped.append(key)
    load_result = module.load_state_dict(matched, strict=False)
    return load_result, skipped


def freeze_backbone(backbone: ProLMBackbone, unfreeze_last_n_layers: int = 0) -> None:
    """Freeze all backbone parameters, then unfreeze the last N encoder layers."""

    for param in backbone.parameters():
        param.requires_grad = False
    if unfreeze_last_n_layers <= 0:
        return
    for layer in backbone.bert.encoder.layer[-unfreeze_last_n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


def trainable_parameters(modules: Iterable[nn.Module]):
    for module in modules:
        yield from (param for param in module.parameters() if param.requires_grad)
