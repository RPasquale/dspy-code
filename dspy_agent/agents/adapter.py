from __future__ import annotations

from typing import Any

from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature


class StrictJSONAdapter(JSONAdapter):
    """A JSON adapter that always uses simple json_object response format.

    DSPy's default JSONAdapter first attempts "structured outputs" (JSON Schema)
    when the provider advertises support, then falls back to json_object with a
    warning if that fails. Some OpenAI-compatible runtimes (e.g., Ollama) do not
    implement structured outputs fully, which leads to noisy warnings.

    This adapter skips the structured-output attempt entirely and always uses
    {"type": "json_object"}, avoiding the warning while preserving robust JSON
    parsing behavior.
    """

    def __init__(self):
        # Disable native function-calling to keep things simple across providers.
        super().__init__(callbacks=None, use_native_function_calling=False)

    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        # Force simple JSON object mode; skip structured outputs completely.
        lm_kwargs = dict(lm_kwargs)
        lm_kwargs["response_format"] = {"type": "json_object"}
        return super(JSONAdapter, self).__call__(lm, lm_kwargs, signature, demos, inputs)

    async def acall(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        # Force simple JSON object mode; skip structured outputs completely.
        lm_kwargs = dict(lm_kwargs)
        lm_kwargs["response_format"] = {"type": "json_object"}
        return await super(JSONAdapter, self).acall(lm, lm_kwargs, signature, demos, inputs)

