from gys.config import ModelConfig
from gys.llm import LLMClient, MockLLM
from gys.reflector import make_reflection_lm, reflection_prompt_template


def test_reflection_template_has_placeholders():
    template = reflection_prompt_template()
    assert "<curr_param>" in template
    assert "<side_info>" in template


def test_reflection_lm_accepts_str_and_messages():
    client = LLMClient(backend=MockLLM(default="new instruction"))
    call = make_reflection_lm(client, ModelConfig())
    assert call("improve this") == "new instruction"
    assert call([{"role": "user", "content": "improve"}]) == "new instruction"
