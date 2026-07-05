from __future__ import annotations

from dataclasses import dataclass

import torch


def cosine_similarity_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_norm = torch.nn.functional.normalize(left, p=2, dim=-1)
    right_norm = torch.nn.functional.normalize(right, p=2, dim=-1)
    return left_norm @ right_norm.T


def cosine_distance_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return 1.0 - cosine_similarity_matrix(left, right)


def mean_pairwise_distance(embeddings: torch.Tensor) -> float:
    if embeddings.shape[0] < 2:
        return 0.0
    distances = cosine_distance_matrix(embeddings, embeddings)
    upper = torch.triu_indices(distances.shape[0], distances.shape[1], offset=1)
    return float(distances[upper[0], upper[1]].mean().item())


@dataclass(frozen=True)
class CreativityComponents:
    plausibility_distance: float
    plausibility_score: float
    novelty: float
    diversity_distance: float
    diversity_score: float
    local_creativity: float


def min_cosine_distances(sample_embeddings: torch.Tensor, reference_embeddings: torch.Tensor) -> torch.Tensor:
    if sample_embeddings.shape[0] == 0:
        raise ValueError("sample_embeddings must contain at least one row.")
    if reference_embeddings.shape[0] == 0:
        raise ValueError("reference_embeddings must contain at least one row.")
    return cosine_distance_matrix(sample_embeddings, reference_embeddings).min(dim=1).values


def normalized_min_cosine_distances(sample_embeddings: torch.Tensor, reference_embeddings: torch.Tensor) -> torch.Tensor:
    return min_cosine_distances(sample_embeddings, reference_embeddings) / 2.0


def normalized_mean_cosine_distances(sample_embeddings: torch.Tensor, reference_embeddings: torch.Tensor) -> torch.Tensor:
    if sample_embeddings.shape[0] == 0:
        raise ValueError("sample_embeddings must contain at least one row.")
    if reference_embeddings.shape[0] == 0:
        raise ValueError("reference_embeddings must contain at least one row.")
    return cosine_distance_matrix(sample_embeddings, reference_embeddings).mean(dim=1) / 2.0


def mean_reference_embedding(reference_embeddings: torch.Tensor) -> torch.Tensor:
    if reference_embeddings.shape[0] == 0:
        raise ValueError("reference_embeddings must contain at least one row.")
    centroid = reference_embeddings.mean(dim=0, keepdim=True)
    return torch.nn.functional.normalize(centroid, p=2, dim=-1)


def centroid_cosine_distances(sample_embeddings: torch.Tensor, reference_embeddings: torch.Tensor) -> torch.Tensor:
    if sample_embeddings.shape[0] == 0:
        raise ValueError("sample_embeddings must contain at least one row.")
    centroid = mean_reference_embedding(reference_embeddings)
    return cosine_distance_matrix(sample_embeddings, centroid).squeeze(1)


def compute_local_creativity(
    plausibility_score: float,
    novelty: float,
    diversity_score: float,
    *,
    alpha: float,
    lambda_penalty: float,
) -> float:
    return alpha * ((plausibility_score ** lambda_penalty) * novelty) + (1.0 - alpha) * diversity_score


def compute_creativity_components(
    sample_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    novelty_scores: torch.Tensor,
    *,
    alpha: float,
    lambda_penalty: float,
) -> CreativityComponents:
    if novelty_scores.shape[0] == 0:
        raise ValueError("novelty_scores must contain at least one row.")
    plausibility_distances = centroid_cosine_distances(sample_embeddings, reference_embeddings)
    plausibility_scores = 1.0 - (plausibility_distances / 2.0)
    diversity_distance = mean_pairwise_distance(sample_embeddings)
    diversity_score = 1.0 - (diversity_distance / 2.0)
    local_creativity = float(
        (
            alpha * ((plausibility_scores**lambda_penalty) * novelty_scores).mean()
            + (1.0 - alpha) * diversity_score
        ).item()
    )
    return CreativityComponents(
        plausibility_distance=float(plausibility_distances.mean().item()),
        plausibility_score=float(plausibility_scores.mean().item()),
        novelty=float(novelty_scores.mean().item()),
        diversity_distance=diversity_distance,
        diversity_score=diversity_score,
        local_creativity=local_creativity,
    )
