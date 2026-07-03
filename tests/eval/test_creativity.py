import math

import torch

from src.eval.creativity import compute_creativity_components, mean_pairwise_distance


def test_mean_pairwise_distance_is_zero_for_single_embedding():
    embeddings = torch.tensor([[1.0, 0.0]])
    assert mean_pairwise_distance(embeddings) == 0.0


def test_compute_creativity_components_uses_centroid_distance_for_plausibility():
    sample_embeddings = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    known_output_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    novelty_scores = torch.tensor([0.2, 0.8], dtype=torch.float32)

    components = compute_creativity_components(
        sample_embeddings,
        known_output_embeddings,
        novelty_scores,
        alpha=0.8,
        lambda_penalty=2.0,
    )

    assert math.isclose(components.plausibility_distance, 0.146446615, rel_tol=1e-6)
    assert math.isclose(components.plausibility_score, 0.926776707, rel_tol=1e-6)
    assert math.isclose(components.novelty, 0.5, rel_tol=1e-6)
    assert math.isclose(components.diversity_distance, 0.292893231, rel_tol=1e-6)
    assert math.isclose(components.diversity_score, 0.853553385, rel_tol=1e-6)
    assert components.local_creativity > 0.0


def test_compute_creativity_components_penalizes_distance_to_candidate_centroid():
    sample_embeddings = torch.tensor([[1.0, 0.0]])
    known_output_embeddings = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    novelty_scores = torch.tensor([1.0])

    components = compute_creativity_components(
        sample_embeddings,
        known_output_embeddings,
        novelty_scores,
        alpha=0.8,
        lambda_penalty=2.0,
    )

    assert math.isclose(components.plausibility_distance, 1.0, rel_tol=1e-6)
    assert math.isclose(components.plausibility_score, 0.5, rel_tol=1e-6)
