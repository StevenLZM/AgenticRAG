# Task 1 report — ModelGateway and structured output validation

## Scope

- Added injected-client `ModelGateway` with plain and schema-validated completion.
- Added immutable `ModelCall` and generic `ModelResponse` metadata envelopes.
- Added strict `RouteDecision` schema plus seven versioned prompt artifacts and a hash loader.

## TDD evidence

- RED: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/runtime/test_model_gateway.py -q`
  failed at collection with `ModuleNotFoundError: No module named 'agentic_rag.models.schemas'` before implementation.
- GREEN: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/runtime/test_model_gateway.py -q`
  passed: `11 passed in 0.03s`.

## Verification

- Retrieval regression: `conda run -n agentic-rag pytest --import-mode=importlib tests/unit/retrieval -q`
  passed: `44 passed in 2.91s`.
- Static checks: `conda run -n agentic-rag mypy src/agentic_rag/runtime/model_gateway.py src/agentic_rag/models/schemas.py`
  passed with no issues.
- Lint and whitespace: Ruff and `git diff --check` both passed.

## Behaviour and provider-shape assumptions

- Client is injected and never constructed by the gateway, so unit tests do not access a network.
- It prefers `client.responses.create(model=..., input=[...])`, normalizes Responses `output_text` (and nested `output` content), and falls back to `client.chat.completions.create(..., messages=[...])` with `choices[0].message.content`.
- It records Responses `usage.input_tokens`/`usage.output_tokens` or Chat `prompt_tokens`/`completion_tokens`; a schema repair aggregates usage from both calls.
- A logical request has one retry owner: gateway retries timeout, connection error, HTTP 429, and HTTP 5xx only, with at most two retries and exponential jitter. Invalid schema triggers exactly one additional repair call; it never exposes a partial parsed value.

## Commit

- `39af5fe feat: add structured model gateway`

## Unresolved items

- `RuntimeConfigSnapshot` currently carries the plan-defined prompt version string. Persisting the per-prompt hash map is a Run persistence responsibility in later Task 8; `load_prompt()` and `prompt_hashes()` supply the immutable hashes for that integration without changing the established snapshot contract prematurely.
