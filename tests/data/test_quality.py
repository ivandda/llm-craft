from src.data.quality import quality_config_from_dict, recipe_quality_reason, score_concept


def test_score_concept_rejects_bad_infinite_craft_noise():
    config = quality_config_from_dict({})

    assert score_concept("bricknado", config).reason == "suspicious_suffix"
    assert score_concept("lava shark", config).keep is True
    assert score_concept("werewolf surfing", config).reason == "gerund_phrase"
    assert score_concept("Titanic 3: Wipeout", config).reason == "title_or_sentence_like"
    assert score_concept("F0I", config).reason == "has_digit"
    assert score_concept("a", config).reason == "short_token"
    assert score_concept("undefined dead lifeform", config).reason == "placeholder_token"


def test_score_concept_keeps_named_entities_and_allowed_acronyms():
    config = quality_config_from_dict({})

    assert score_concept("messi", config).keep is True
    assert score_concept("star wars", config).keep is True
    assert score_concept("usa", config).keep is True


def test_recipe_quality_rejects_identity_outputs():
    config = quality_config_from_dict({})

    reason = recipe_quality_reason("fire", "water", "fire", config)

    assert reason == "identity"
