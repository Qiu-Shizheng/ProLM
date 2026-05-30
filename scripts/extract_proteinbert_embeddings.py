"""Extract per-protein sequence embeddings with ProteinBERT.

This script is optional. It requires the third-party ProteinBERT package and a
downloaded ProteinBERT checkpoint. The generated .npz file can be passed to
`python -m prolm.finetune --protein-embeddings`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract global ProteinBERT embeddings from protein sequences.")
    parser.add_argument("--sequences-csv", required=True, help="CSV with Protein and Sequence columns.")
    parser.add_argument("--output-npz", required=True, help="Output .npz file.")
    parser.add_argument("--excluded-output", default=None, help="Optional text file for sequences exceeding max length.")
    parser.add_argument("--proteinbert-dir", required=True, help="Directory containing the ProteinBERT model dump.")
    parser.add_argument("--proteinbert-checkpoint", required=True, help="ProteinBERT checkpoint file name or path.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from proteinbert import load_pretrained_model
        from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
    except ImportError as exc:
        raise SystemExit(
            "ProteinBERT is not installed. Install it in a compatible TensorFlow/Keras environment first."
        ) from exc

    df = pd.read_csv(args.sequences_csv)
    required = {"Protein", "Sequence"}
    if not required.issubset(df.columns):
        raise ValueError("Sequence CSV must contain Protein and Sequence columns.")

    ids: list[str] = []
    seqs: list[str] = []
    excluded: list[str] = []
    for row in df.itertuples(index=False):
        protein = str(getattr(row, "Protein"))
        sequence = str(getattr(row, "Sequence"))
        if len(sequence) > args.max_seq_len:
            excluded.append(protein)
            continue
        ids.append(protein)
        seqs.append(sequence)

    if args.excluded_output:
        Path(args.excluded_output).write_text("\n".join(excluded) + ("\n" if excluded else ""), encoding="utf-8")

    pretrained_model_generator, input_encoder = load_pretrained_model(
        local_model_dump_dir=args.proteinbert_dir,
        local_model_dump_file_name=args.proteinbert_checkpoint,
    )
    model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(args.max_seq_len))

    features = {}
    for start in tqdm(range(0, len(ids), args.batch_size), desc="ProteinBERT"):
        batch_ids = ids[start : start + args.batch_size]
        batch_seqs = seqs[start : start + args.batch_size]
        encoded_x = input_encoder.encode_X(batch_seqs, args.max_seq_len)
        _, sequence_embeddings = model.predict(encoded_x, batch_size=args.batch_size, verbose=0)
        for i, protein in enumerate(batch_ids):
            features[protein] = sequence_embeddings[i].astype(np.float32)

    np.savez(args.output_npz, **features)
    print(f"Saved {len(features)} embeddings to {args.output_npz}")


if __name__ == "__main__":
    main()
