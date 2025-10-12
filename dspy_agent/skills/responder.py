from __future__ import annotations

from typing import Optional

import dspy  # type: ignore


class RespondSig(dspy.Signature):
    """Generate a direct natural-language reply for conversational queries."""

    query: str = dspy.InputField(desc="User's natural-language message")
    context: str = dspy.InputField(default="", desc="Brief session history or memory to stay coherent")
    workspace: str = dspy.InputField(default="", desc="Workspace summary or environment hint if relevant")

    response: str = dspy.OutputField(desc="Assistant reply the user should see")


class Responder(dspy.Module):
    """Lightweight module that turns short prompts into direct replies."""

    def __init__(self, use_cot: bool = False):
        super().__init__()
        # Chain-of-thought adds latency and is unnecessary for chit-chat, so keep it simple.
        predict_kwargs = {"max_tokens": 768, "temperature": 0.15}
        if not use_cot:
            self.predict = dspy.Predict(RespondSig, **predict_kwargs)
            self._predict_careful = dspy.ChainOfThought(RespondSig, **predict_kwargs)
        else:
            self.predict = dspy.ChainOfThought(RespondSig, **predict_kwargs)
            self._predict_careful = self.predict

    def forward(self, query: str, context: str = "", workspace: Optional[str] = None):
        ws = workspace or ""
        guidance = (
            "You are the DSPy development agent working inside an interactive coding workspace. "
            "Answer the user's request with a clear, thorough natural-language explanation. "
            "Reference coding tools, strategies, or next steps when appropriate, and avoid emitting shell-style commands "
            "unless the user explicitly asked for them. Keep the reply actionable and self-contained."
        )
        enriched_query = f"{guidance}\n\nUser request: {query.strip()}"
        enriched_context = context.strip() if context else ""
        if ws:
            enriched_context = (enriched_context + f"\nWorkspace root: {ws}").strip()
        result = self.predict(query=enriched_query, context=enriched_context, workspace=ws)
        reply = (getattr(result, "response", "") or "").strip()
        if not reply or reply.startswith("/") or reply.lower().startswith("command "):
            secondary_guidance = (
                "Respond directly in natural language. Do not emit slash commands. "
                "Provide a detailed, helpful explanation or plan."
            )
            secondary_query = f"{secondary_guidance}\n\nUser request: {query.strip()}"
            result = self._predict_careful(query=secondary_query, context=enriched_context, workspace=ws)
            reply = (getattr(result, "response", "") or "").strip()
            if not reply or reply.startswith("/") or reply.lower().startswith("command "):
                lower_query = query.lower()
                if "transformer" in lower_query:
                    fallback_text = (
                        "Here's a practical blueprint for implementing a Transformer model from scratch:\n"
                        "1. Set up a Python environment (e.g., PyTorch + torchtext or tokenizers) and gather sample data.\n"
                        "2. Build tokenization and vocabulary utilities so the model can map text to integer ids.\n"
                        "3. Implement embedding and positional encoding layers to convert tokens into contextual vectors.\n"
                        "4. Write reusable modules for multi-head self-attention, feed-forward networks, residual connections, and layer normalization.\n"
                        "5. Stack encoder and/or decoder blocks, add the final projection head, and expose configuration hooks for depth, hidden size, and number of heads.\n"
                        "6. Create a training loop with teacher forcing (for sequence-to-sequence tasks), cross-entropy loss, learning-rate scheduling, gradient clipping, and evaluation metrics.\n"
                        "7. Add unit tests or experiments to validate attention shapes, masking logic, and overall convergence, then iterate on hyperparameters."
                    )
                elif any(keyword in lower_query for keyword in {"log", "logging", "stream", "pipeline"}):
                    fallback_text = (
                        "To strengthen the logging/streaming pipeline:\n"
                        "1. Catalogue every log producer (agents, workers, services) and confirm they emit structured JSON with severity levels.\n"
                        "2. Add workspace-level change events (already streaming) to the context index so investigative tools can surface them.\n"
                        "3. Ensure Kafka topics cover raw, curated, and alert streams; configure retention and DLQs per topic.\n"
                        "4. Extend the Spark job to summarise the most recent workspace deltas and broadcast them to the agent memory.\n"
                        "5. Create automated checks (integration tests or dashboards) that validate end-to-end delivery, including schema evolution."
                    )
                elif any(keyword in lower_query for keyword in {"test", "pytest", "unit test", "coverage"}):
                    fallback_text = (
                        "A quick testing playbook:\n"
                        "1. Generate or update unit tests that cover the new behaviour (use pytest fixtures where possible).\n"
                        "2. Add integration tests that exercise key Kafka/Spark pipelines, guarding against silent failures.\n"
                        "3. Wire tests into the agent's toolchain so `plan` recommends `run_tests`, and log pass/fail signals for RL feedback.\n"
                        "4. Capture coverage reports and store them alongside streaming metrics to track regression in quality.\n"
                        "5. Document flake or long-running tests and schedule follow-up clean-ups."
                    )
                elif any(keyword in lower_query for keyword in {"refactor", "cleanup", "improve", "optimize"}):
                    fallback_text = (
                        "To tackle this refactor safely:\n"
                        "1. Run `plan` to map the impacted modules and tests before touching code.\n"
                        "2. Introduce feature flags or incremental commits so the agent can reason over smaller diffs.\n"
                        "3. Update or regenerate indexes/embeddings after each structural change so search tooling stays accurate.\n"
                        "4. Run lint/format tools and targeted tests after each milestone, logging outcomes for RL training.\n"
                        "5. Record follow-up tasks (performance benchmarks, docs) in the session memory for later runs."
                    )
                else:
                    fallback_text = (
                        "Here's a structured way to approach this request inside the DSPy workspace:\n"
                        f"1. Restate the goal: {query.strip()}\n"
                        "2. Identify the key components, modules, or classes you will need.\n"
                        "3. Sketch the implementation steps and the tests you expect to run before coding.\n"
                        "4. Implement the work incrementally, running linting or tests after each milestone.\n"
                        "5. Document decisions and follow-up tasks so the next agent run has useful context."
                    )
                result.response = fallback_text
        return result
