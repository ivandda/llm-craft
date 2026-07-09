"""Clusteriza los embeddings de las racionalizaciones e imprime los candidatos
más centrales de cada cluster, para inspección manual. Los grupos salen por dominio
semántico (silhouette ~0.03); ver FINDINGS.md.

Reusa runs/analysis/rationale_emb.npy si existe (mismo orden que
candidates_loader.load_candidates_jsonl); si no, recomputa los embeddings.

    uv run python -m src.analysis.clustering.inspect_rationale_clusters --k 10
"""
import argparse

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..candidates_loader import load_candidates_jsonl

DEFAULT_EMB_PATH = "runs/analysis/rationale_emb.npy"


def load_or_compute_rationale_embeddings(rationales: list[str], emb_path: str, use_fake: bool) -> np.ndarray:
    """Carga el .npy cacheado si coincide en cantidad; si no, lo recomputa."""
    try:
        emb = np.load(emb_path)
        if emb.shape[0] == len(rationales):
            print(f"      usando embeddings cacheados de {emb_path} {emb.shape}")
            return emb
        print(f"      {emb_path} tiene {emb.shape[0]} filas != {len(rationales)}; recomputando")
    except FileNotFoundError:
        print(f"      {emb_path} no existe; computando embeddings de racionalizaciones")

    from ..embed_utils import Embedder, FakeEmbedder

    embedder = FakeEmbedder() if use_fake else Embedder()
    emb = embedder.encode(rationales)
    np.save(emb_path, emb)
    print(f"      embeddings guardados en {emb_path} {emb.shape}")
    return emb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_jsonl", default="datasets/final-10k/train.jsonl")
    parser.add_argument("--emb_path", default=DEFAULT_EMB_PATH, help="Cache de embeddings de racionalizaciones.")
    parser.add_argument("--k", type=int, default=10, help="Cantidad de clusters (KMeans).")
    parser.add_argument("--top_n", type=int, default=10, help="Ejemplos centrales a mostrar por cluster.")
    parser.add_argument("--output_csv", default="runs/analysis/rationale_clusters_inspection.csv")
    parser.add_argument("--fake_embeddings", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("[1/3] Cargando candidatos...")
    df = load_candidates_jsonl(args.train_jsonl).reset_index(drop=True)
    rationales = df["rationale"].fillna("").astype(str).tolist()
    print(f"      {len(df)} candidatos")

    print("[2/3] Embeddings de racionalizaciones...")
    emb = load_or_compute_rationale_embeddings(rationales, args.emb_path, args.fake_embeddings)
    assert emb.shape[0] == len(df), f"desalineado: {emb.shape[0]} emb vs {len(df)} candidatos"
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)

    print(f"[3/3] KMeans (k={args.k})...")
    km = KMeans(n_clusters=args.k, n_init=4, random_state=args.seed)
    labels = km.fit_predict(emb)

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(emb), size=min(3000, len(emb)), replace=False)
    sil = silhouette_score(emb[idx], labels[idx])
    print(f"      silhouette (muestra {len(idx)}) = {sil:.3f}\n")

    dist = np.linalg.norm(emb - km.cluster_centers_[labels], axis=1)
    out = df.copy()
    out["cluster_id"] = labels
    out["dist_to_centroid"] = dist
    out.sort_values(["cluster_id", "dist_to_centroid"]).to_csv(args.output_csv, index=False)

    # Vista en consola: los más centrales de cada cluster, de mayor a menor tamaño.
    for cid in np.argsort(-np.bincount(labels, minlength=args.k)):
        members = np.where(labels == cid)[0]
        members = members[np.argsort(dist[members])][: args.top_n]
        print(f"=== Cluster {cid}  ({(labels == cid).sum()} candidatos) ===")
        for i in members:
            row = df.iloc[i]
            print(f"  {row['a']} + {row['b']} -> {row['c']}")
            print(f"      · {str(row['rationale'])[:130]}")
        print()

    print(f"CSV completo en: {args.output_csv}")


if __name__ == "__main__":
    main()
