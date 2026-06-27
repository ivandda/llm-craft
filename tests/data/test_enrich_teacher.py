from src.data.enrich_teacher import (
    StructuredEnrichmentConfig,
    TeacherCallError,
    TeacherCandidate,
    TeacherEnrichment,
    TeacherResult,
    TokenUsage,
    build_teacher_prompt,
    enrich_recipe,
    enrich_split,
    estimate_cost,
    load_recipes,
)


class FakeStructuredTeacher:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def generate(self, input_a, input_b, observed_outputs, max_alternatives):
        self.calls.append(
            {
                "input_a": input_a,
                "input_b": input_b,
                "observed_outputs": list(observed_outputs),
                "max_alternatives": max_alternatives,
            }
        )
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def teacher_result(
    candidate_outputs=None,
    keep_recipe=True,
    reject_reason=None,
    usage=None,
):
    return TeacherResult(
        enrichment=TeacherEnrichment(
            keep_recipe=keep_recipe,
            reject_reason=reject_reason,
            candidate_outputs=candidate_outputs or [],
        ),
        usage=usage or TokenUsage(input_tokens=100, output_tokens=40, total_tokens=140),
    )


def candidate(output, rationale="short rationale", source="teacher"):
    return TeacherCandidate(output=output, rationale=rationale, source=source)


def test_enrich_recipe_accepts_partial_outputs_without_forcing_alternatives():
    teacher = FakeStructuredTeacher(
        [
            teacher_result(
                candidate_outputs=[
                    candidate("steam", "Fire heats water until it becomes steam.", "observed"),
                ],
            )
        ]
    )

    record, rejection, usage = enrich_recipe(
        recipe={"input_a": "Fire", "input_b": "Water", "outputs": ["Steam"]},
        split="train",
        teacher=teacher,
        config=StructuredEnrichmentConfig(target_num_outputs=5),
    )

    assert rejection is None
    assert usage == TokenUsage(input_tokens=100, output_tokens=40, total_tokens=140)
    assert record["quality_status"] == "partial_enrichment"
    assert record["candidate_outputs"] == [
        {
            "output": "steam",
            "source": "observed",
            "rationale": "Fire heats water until it becomes steam.",
            "rank": 1,
        }
    ]
    assert teacher.calls[0]["max_alternatives"] == 4


def test_enrich_recipe_filters_invalid_teacher_alternatives_programmatically():
    teacher = FakeStructuredTeacher(
        [
            teacher_result(
                candidate_outputs=[
                    candidate("steam", source="observed"),
                    candidate("steam", source="teacher"),
                    candidate("fire"),
                    candidate("123"),
                    candidate("hot spring", "Heat and water suggest a hot spring."),
                ],
            )
        ]
    )

    record, rejection, _usage = enrich_recipe(
        recipe={"input_a": "fire", "input_b": "water", "outputs": ["steam"]},
        split="dev",
        teacher=teacher,
        config=StructuredEnrichmentConfig(target_num_outputs=5),
    )

    assert rejection is None
    assert record["candidate_outputs"] == [
        {"output": "steam", "source": "observed", "rationale": "short rationale", "rank": 1},
        {
            "output": "hot spring",
            "source": "teacher",
            "rationale": "Heat and water suggest a hot spring.",
            "rank": 2,
        },
    ]


def test_enrich_recipe_preserves_teacher_global_order_with_programmatic_ranks():
    teacher = FakeStructuredTeacher(
        [
            teacher_result(
                candidate_outputs=[
                    candidate("hot spring", "Heat and water suggest a hot spring.", "teacher"),
                    candidate("steam", "Fire heats water until it becomes steam.", "observed"),
                ],
            )
        ]
    )

    record, rejection, _usage = enrich_recipe(
        recipe={"input_a": "fire", "input_b": "water", "outputs": ["steam"]},
        split="dev",
        teacher=teacher,
        config=StructuredEnrichmentConfig(target_num_outputs=5),
    )

    assert rejection is None
    assert record["candidate_outputs"] == [
        {
            "output": "hot spring",
            "source": "teacher",
            "rationale": "Heat and water suggest a hot spring.",
            "rank": 1,
        },
        {
            "output": "steam",
            "source": "observed",
            "rationale": "Fire heats water until it becomes steam.",
            "rank": 2,
        },
    ]


def test_enrich_recipe_dedupes_compact_spelling_variants():
    teacher = FakeStructuredTeacher(
        [
            teacher_result(
                candidate_outputs=[
                    candidate("sand castle", "A castle made of sand.", "observed"),
                    candidate("sandcastle", "Same compact spelling.", "observed"),
                    candidate("desert castle", "A castle in sandy terrain.", "teacher"),
                ],
            )
        ]
    )

    record, rejection, _usage = enrich_recipe(
        recipe={"input_a": "castle", "input_b": "sand", "outputs": ["sand castle", "sandcastle"]},
        split="dev",
        teacher=teacher,
        config=StructuredEnrichmentConfig(target_num_outputs=5),
    )

    assert rejection is None
    assert [candidate["output"] for candidate in record["candidate_outputs"]] == [
        "sand castle",
        "desert castle",
    ]
    assert [candidate["rank"] for candidate in record["candidate_outputs"]] == [1, 2]


def test_enrich_recipe_rejects_when_teacher_rejects_whole_recipe():
    teacher = FakeStructuredTeacher(
        [
            teacher_result(
                keep_recipe=False,
                reject_reason="clearly unrelated",
            )
        ]
    )

    record, rejection, _usage = enrich_recipe(
        recipe={"input_a": "fire", "input_b": "water", "outputs": ["random word"]},
        split="test",
        teacher=teacher,
        config=StructuredEnrichmentConfig(),
    )

    assert record is None
    assert rejection["reject_reason"] == "clearly_unrelated"
    assert rejection["outputs"] == ["random word"]


def test_enrich_recipe_retries_teacher_response_errors():
    teacher = FakeStructuredTeacher(
        [
            TeacherCallError("bad json"),
            teacher_result(candidate_outputs=[candidate("steam", source="observed")]),
        ]
    )

    record, rejection, usage = enrich_recipe(
        recipe={"input_a": "fire", "input_b": "water", "outputs": ["steam"]},
        split="train",
        teacher=teacher,
        config=StructuredEnrichmentConfig(max_retries=1),
    )

    assert rejection is None
    assert record["candidate_outputs"][0]["output"] == "steam"
    assert usage == TokenUsage(input_tokens=100, output_tokens=40, total_tokens=140)
    assert len(teacher.calls) == 2


def test_enrich_split_resume_skips_only_records_with_candidates(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    input_path.write_text(
        '{"input_a": "fire", "input_b": "water", "outputs": ["steam"]}\n',
        encoding="utf-8",
    )
    output_path.write_text(
        '{"input_a": "fire", "input_b": "water", "candidate_outputs": []}\n',
        encoding="utf-8",
    )
    teacher = FakeStructuredTeacher([teacher_result(candidate_outputs=[candidate("steam", source="observed")])])

    counts = enrich_split(
        input_path=input_path,
        output_path=output_path,
        rejected_path=rejected_path,
        split="train",
        teacher=teacher,
        config=StructuredEnrichmentConfig(),
        limit=None,
        resume=True,
    )

    assert counts["written"] == 1
    assert counts["skipped_existing"] == 0
    assert len(teacher.calls) == 1


def test_estimate_cost_uses_flash_lite_default_prices():
    cost = estimate_cost(
        TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000, total_tokens=2_000_000),
        StructuredEnrichmentConfig(),
    )

    assert cost == 0.50


def test_load_recipes_uses_seeded_random_sample(tmp_path):
    input_path = tmp_path / "recipes.jsonl"
    input_path.write_text(
        "\n".join(
            f'{{"input_a": "a{i}", "input_b": "b{i}", "outputs": ["o{i}"]}}'
            for i in range(20)
        )
        + "\n",
        encoding="utf-8",
    )

    first_sample = load_recipes(input_path, sample_size=5, seed=7)
    second_sample = load_recipes(input_path, sample_size=5, seed=7)

    assert first_sample == second_sample
    assert len(first_sample) == 5
    assert first_sample != load_recipes(input_path, sample_size=None, seed=7)[:5]


def test_strict_prompt_includes_rejection_decision_process():
    prompt = build_teacher_prompt(
        input_a="cat",
        input_b="ocean",
        observed_outputs=["fish"],
        max_alternatives=4,
        max_rationale_words=24,
        prompt_style="strict",
    )

    assert "Use this decision process" in prompt
    assert "If unsure, omit the output" in prompt
    assert "Do not use numeric scores" in prompt
    assert "cat + ocean -> fish" in prompt
