"""Command-line fine-tuning for binary classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data import ProteinClassificationDataset, load_prolm_data
from .model import (
    ProLMBackbone,
    ProLMClassifier,
    freeze_backbone,
    infer_checkpoint_shape,
    load_matching_state_dict,
    strip_module_prefix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ProLM on a binary disease classification task.")
    parser.add_argument("--expression-csv", required=True, help="CSV with eid plus protein expression columns.")
    parser.add_argument("--labels-csv", required=True, help="CSV with eid and binary label columns.")
    parser.add_argument("--protein-embeddings", required=True, help="Protein sequence embeddings in .npz format.")
    parser.add_argument("--correlation-csv", default=None, help="Optional protein correlation matrix CSV.")
    parser.add_argument("--pretrained-checkpoint", default=None, help="Optional ProLM checkpoint .pt file.")
    parser.add_argument("--output-dir", default="outputs/prolm_finetune", help="Directory for outputs.")
    parser.add_argument("--eid-col", default="eid")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--hidden-size", type=int, default=64, help="Used only when no checkpoint is supplied.")
    parser.add_argument("--num-layers", type=int, default=2, help="Used only when no checkpoint is supplied.")
    parser.add_argument("--chunk-size", type=int, default=64, help="Used only when no checkpoint is supplied.")
    parser.add_argument("--unfreeze-last-n-layers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Training device.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def build_model(args: argparse.Namespace, data, device: torch.device) -> ProLMClassifier:
    feature_dim = int(data.features_tensor.shape[1])
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    chunk_size = args.chunk_size
    state_dict = None
    if args.pretrained_checkpoint:
        state_dict = torch.load(args.pretrained_checkpoint, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = strip_module_prefix(state_dict)
        shape = infer_checkpoint_shape(state_dict)
        feature_dim = shape.feature_dim
        hidden_size = shape.hidden_size
        num_layers = shape.num_layers
        chunk_size = shape.chunk_size
        if data.features_tensor.shape[1] != feature_dim:
            raise ValueError(
                "Embedding dimension does not match checkpoint: "
                f"data has {data.features_tensor.shape[1]}, checkpoint expects {feature_dim}."
            )

    backbone = ProLMBackbone(
        feature_dim=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_proteins=len(data.protein_names),
        features_tensor=data.features_tensor,
        bias_matrix=data.bias_matrix,
        chunk_size=chunk_size,
    )
    if state_dict is not None:
        load_result, skipped = load_matching_state_dict(
            backbone,
            state_dict,
            skip_keys={"features_tensor", "bias_matrix_full"},
        )
        if skipped:
            print(f"Skipped {len(skipped)} checkpoint tensors with incompatible names or shapes.")
        if load_result.missing_keys:
            print(f"Missing after partial checkpoint load: {load_result.missing_keys[:5]}")
        if load_result.unexpected_keys:
            print(f"Unexpected after partial checkpoint load: {load_result.unexpected_keys[:5]}")
    freeze_backbone(backbone, args.unfreeze_last_n_layers)
    model = ProLMClassifier(backbone, num_classes=2).to(device)
    return model


def evaluate(model: ProLMClassifier, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    y_true = []
    y_prob = []
    with torch.no_grad():
        for x, y, _ in loader:
            logits = model(x.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(y.numpy().tolist())
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }
    metrics["auroc"] = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan")
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available. Use --device cpu to run on CPU.")
    device = torch.device(args.device)

    data = load_prolm_data(
        args.expression_csv,
        args.labels_csv,
        args.protein_embeddings,
        args.correlation_csv,
        eid_col=args.eid_col,
        label_col=args.label_col,
    )
    dataset = ProteinClassificationDataset(data.expression, data.labels)
    labels = data.labels["label"].to_numpy()
    indices = np.arange(len(dataset))
    stratify = labels if len(np.unique(labels)) == 2 and min(np.bincount(labels)) >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    model = build_model(args, data, device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_auroc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = evaluate(model, test_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(losses))
        history.append(metrics)
        print(json.dumps(metrics, indent=2))
        score = metrics["auroc"] if not np.isnan(metrics["auroc"]) else metrics["accuracy"]
        if score > best_auroc:
            best_auroc = score
            torch.save(model.state_dict(), output_dir / "best_classifier.pt")

    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "protein_order.txt").write_text("\n".join(data.protein_names) + "\n", encoding="utf-8")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
