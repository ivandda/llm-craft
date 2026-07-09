"""Diagnóstico del clustering de vectores de relación y de racionalizaciones.

Usa MiniBatchKMeans + silhouette sobre una muestra para escalar a ~37k tripletas.
Imprime dos bloques:
  REL: silhouette del vector de relación r=c-(a+b)/2 (diff_mean) y r=[c-a,c-b]
       (diff_concat), más Spearman(rank, dist_to_centroid).
  RATIONALE: silhouette del clustering de las racionalizaciones + ejemplos por cluster.

    uv run python -m src.analysis.clustering.validate --train_jsonl datasets/final-10k/train.jsonl --model glove
"""
import argparse
import collections

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from . import relation_vectors as rv
from ..candidates_loader import load_candidates_jsonl
from ..embedder_registry import MODELS, build_embedder


def _sampled_silhouette(X: np.ndarray, labels: np.ndarray, n: int = 4000, seed: int = 0) -> float:
    if len(X) <= n:
        return float(silhouette_score(X, labels))
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    return float(silhouette_score(X[idx], labels[idx]))


def _dist_to_centroid(r: np.ndarray, k: int, seed: int = 42):
    km = MiniBatchKMeans(n_clusters=k, n_init=5, random_state=seed).fit(r)
    lab = km.predict(r)
    d = np.linalg.norm(r - km.cluster_centers_[lab], axis=1)
    return lab, d, km


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--k_rel", type=int, default=20, help="k para el clustering de relación")
    parser.add_argument("--k_rat", type=int, default=25, help="k para el clustering de rationale")
    parser.add_argument("--top_n", type=int, default=6, help="ejemplos centrales por cluster de rationale")
    parser.add_argument("--n_clusters_show", type=int, default=8)
    parser.add_argument("--model", default="bge-base", choices=list(MODELS.keys()),
                        help="modelo de embeddings (default bge-base contextual; 'glove' = estático)")
    parser.add_argument("--fake_embeddings", action="store_true")
    args = parser.parse_args()

    print("[1/3] Cargando datos...")
    df = load_candidates_jsonl(args.train_jsonl).reset_index(drop=True)
    rank = df["rank"].values
    print(f"      {len(df)} tripletas, {df[['a','b']].drop_duplicates().shape[0]} pares")

    print(f"[2/3] Embeddings (modelo={args.model}{' FAKE' if args.fake_embeddings else ''})...")
    embedder, prefix = build_embedder(args.model, args.fake_embeddings)
    ea = embedder.encode(df["a"].tolist(), prefix=prefix, show_progress=False)
    eb = embedder.encode(df["b"].tolist(), prefix=prefix, show_progress=False)
    ec = embedder.encode(df["c"].tolist(), prefix=prefix, show_progress=False)
    erat = embedder.encode(df["rationale"].tolist(), prefix=prefix, show_progress=False)
    if hasattr(embedder, "oov_report"):
        print("      " + embedder.oov_report())

    print("\n[3/3] Diagnósticos")
    print("=" * 70)
    print("REL (vector de relación): se esperaba Spearman(rank,dist) POSITIVO y signif.")
    print("=" * 70)
    for pool in ("diff_mean", "diff_concat"):
        r = rv.l2_normalize(rv.compute_relation_vectors(ea, eb, ec, strategy=pool))
        lab, d, _ = _dist_to_centroid(r, args.k_rel)
        rho, p = spearmanr(rank, d)
        sil = _sampled_silhouette(r, lab)
        print(f"  {pool:11s} k={args.k_rel}  Spearman(rank,dist)={rho:+.3f} (p={p:.1e})  silhouette={sil:.3f}")

    print("\n" + "=" * 70)
    print("RATIONALE (clustering de las explicaciones): agrupa por dominio/tema")
    print("=" * 70)
    km = MiniBatchKMeans(n_clusters=args.k_rat, n_init=5, random_state=42).fit(erat)
    lab = km.predict(erat)
    d = np.linalg.norm(erat - km.cluster_centers_[lab], axis=1)
    print(f"  k={args.k_rat}  silhouette={_sampled_silhouette(erat, lab):.3f}")
    sizes = collections.Counter(lab)
    for cid, _ in sizes.most_common(args.n_clusters_show):
        idx = np.where(lab == cid)[0]
        idx = idx[np.argsort(d[idx])][: args.top_n]
        print(f"\n  --- cluster {cid} (n={sizes[cid]}) ---")
        for i in idx:
            row = df.iloc[i]
            print(f"    {row['a']} + {row['b']} -> {row['c']}  | {row['rationale'][:68]}")

    print("\nConclusión (ver FINDINGS.md): el vector de relación no muestra señal")
    print("(Spearman ~0, silhouette ~0); el clustering de rationale sí da dominios")
    print("coherentes, pero por TEMA, no por tipo-de-regla composicional.")


if __name__ == "__main__":
    main()
