# adapters/llm

Concrete `ILLMProvider` implementations, one module per provider.

## Files

- `__init__.py` — exports all provider adapters.
- `openai.py` — `OpenAIAdapter`: calls the OpenAI chat completions API.

Add one module per additional LLM provider (e.g. `anthropic.py`), each implementing `ILLMProvider` from `core/protocols/llm.py`.
