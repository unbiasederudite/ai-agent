"""Protocol interfaces for all external providers, tools, and strategies."""

from ai_agent.core.protocols.agent import IAgent as IAgent
from ai_agent.core.protocols.llm import ILLMProvider as ILLMProvider
from ai_agent.core.protocols.strategy import IReasoningStrategy as IReasoningStrategy
from ai_agent.core.protocols.tool import ITool as ITool

__all__ = ["IAgent", "ILLMProvider", "IReasoningStrategy", "ITool"]
