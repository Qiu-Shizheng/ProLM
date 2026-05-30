"""Generate example input files for ProLM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ProLM input files.")
    parser.add_argument("--output-dir", default="examples/synthetic")
    parser.add_argument("--n-samples", type=int, default=48)
    parser.add_argument("--n-proteins", type=int, default=24)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    protein_order = [
        "a1bg", "aamdc", "aarsd1", "abca2", "abhd14b", "abl1", "abo", "abraxas2",
        "acaa1", "acadm", "acadsb", "acan", "ace", "ace2", "ache", "acot13",
        "acox1", "acp1", "acp5", "acp6", "acrbp", "acrv1", "acsl1", "acta2",
        "actn2", "actn4", "acvrl1", "acy1", "acy3", "acyp1", "ada", "ada2",
        "adam12", "adam15", "adam22", "adam23", "adam8", "adam9", "adamts1",
        "adamts13", "adamts15", "adamts16", "adamts4", "adamts8", "adamtsl2",
        "adamtsl4", "adamtsl5", "adcyap1r1", "add1", "adgrb3", "adgrd1",
        "adgre1", "adgre2", "adgre5", "adgrf5", "adgrg1", "adgrg2", "adgrv1",
        "adh1b", "adh4", "adipoq", "adm", "adra2a", "afap1", "afm", "afp",
        "agbl2", "ager", "agr2", "agr3", "agrn", "agrp", "agt", "agxt",
        "ahcy", "ahnak", "ahnak2", "ahsa1", "ahsg", "ahsp", "aida", "aif1",
        "aif1l", "aifm1", "ak1", "ak2", "akap12", "akr1b1", "akr1b10",
        "akr1c4", "akr7l", "akt1s1", "akt2", "akt3", "alcam", "aldh1a1",
    ]
    if args.n_proteins > len(protein_order):
        raise ValueError(f"n-proteins can be at most {len(protein_order)} for this example generator.")
    proteins = protein_order[: args.n_proteins]
    eids = [str(i) for i in range(1, args.n_samples + 1)]

    latent = rng.normal(size=(args.n_samples, 3))
    loadings = rng.normal(size=(3, args.n_proteins))
    expression = latent @ loadings + rng.normal(scale=0.7, size=(args.n_samples, args.n_proteins))
    expression = (expression - expression.mean(axis=0)) / expression.std(axis=0)

    signal = expression[:, : min(4, args.n_proteins)].mean(axis=1) + rng.normal(scale=0.25, size=args.n_samples)
    labels = (signal > np.median(signal)).astype(int)

    expression_df = pd.DataFrame(expression, columns=proteins)
    expression_df.insert(0, "eid", eids)
    expression_df.to_csv(out / "expression.csv", index=False)
    pd.DataFrame({"eid": eids, "label": labels}).to_csv(out / "labels.csv", index=False)

    features = {
        protein: rng.normal(size=args.feature_dim).astype(np.float32)
        for protein in proteins
    }
    np.savez(out / "protein_embeddings.npz", **features)

    corr = np.corrcoef(expression, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    corr_df = pd.DataFrame(corr, index=proteins, columns=proteins)
    corr_df.to_csv(out / "protein_correlation.csv")

    sequences = []
    alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    for protein in proteins:
        length = int(rng.integers(80, 180))
        sequences.append({"Protein": protein, "Sequence": "".join(rng.choice(alphabet, size=length))})
    pd.DataFrame(sequences).to_csv(out / "sequences.csv", index=False)

    print(f"Synthetic ProLM inputs written to {out}")


if __name__ == "__main__":
    main()
