from .litellm_models import LiteLLMModel

__all__ = ["LiteLLMModel"]


def get_model(model: str, system_prompt: str = None) -> LiteLLMModel:
    return LiteLLMModel(model, system_prompt)
