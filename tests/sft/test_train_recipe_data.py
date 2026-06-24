from src.sft.train import recipe_outputs, render_recipe_prompt


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"user:{messages[0]['content']}\nassistant:"


def test_recipe_outputs_supports_multi_output_records():
    assert recipe_outputs({"outputs": ["steam", "mist"]}) == ["steam", "mist"]


def test_render_recipe_prompt_injects_prompt_at_runtime():
    prompt = render_recipe_prompt(
        input_a="fire",
        input_b="water",
        prompt_template="{input_a}+{input_b}=?",
        tokenizer=FakeTokenizer(),
    )

    assert prompt == "user:fire+water=?\nassistant:"
