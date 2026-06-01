"""ConversationFactory: builds a fully wired Conversation from ConversationConfig."""

from __future__ import annotations

from typing import Final

from ai_agent.core.services.conversation import Conversation
from ai_agent.core.exceptions import ConfigError
from ai_agent.core.factories.agent import AgentFactory
from ai_agent.core.factories.strategy import StrategyFactory
from ai_agent.core.models.budget import ContextBudget
from ai_agent.core.models.config import ConversationConfig
from ai_agent.core.models.llm import LLM
from ai_agent.core.models.tool import Tool
from ai_agent.core.protocols.llm import ILLMProvider
from ai_agent.core.registries.agent import AgentRegistry
from ai_agent.core.registries.llm import LLMRegistry
from ai_agent.core.registries.tool import ToolRegistry
from ai_agent.core.services.compaction import CompactionService
from ai_agent.core.services.tool import ToolService
from ai_agent.core.strategies.base import BaseStrategy
from ai_agent.core.tools.base import BaseTool


class ConversationFactory:
    """Builds a fully wired Conversation from ConversationConfig."""

    def __init__(
        self,
        llm_implementations: dict[str, ILLMProvider],
        tool_implementations: dict[str, type[BaseTool]],
        strategy_implementations: dict[str, type[BaseStrategy]],
    ) -> None:
        self._llm_implementations: Final = llm_implementations
        self._tool_implementations: Final = tool_implementations
        self._strategy_implementations: Final = strategy_implementations

    def build(self, config: ConversationConfig) -> Conversation:
        """Build a fully wired Conversation.

        Args:
            config: Conversation configuration.

        Returns:
            A ready-to-use Conversation instance.

        Raises:
            ConfigError: Unknown provider or tool type.
        """
        # 1. registries
        llm_registry = self._build_llm_registry(config)
        tool_registry = self._build_tool_registry(config)

        # 2. services
        tool_service = ToolService(tool_registry)
        compaction_provider = llm_registry.resolve_implementation(config.compaction.llm.provider)
        compaction_service = CompactionService(
            provider=compaction_provider,
            model=config.compaction.llm.model,
            settings=config.compaction.settings,
        )

        # 3. factories
        strategy_factory = StrategyFactory(
            implementations=self._strategy_implementations,
            tool_service=tool_service,
        )
        agent_factory = AgentFactory(strategy_factory=strategy_factory)

        # 4. agent registry
        agent_registry = self._build_agent_registry(config, agent_factory)

        # 5. default run settings + context budget
        default_agent = agent_registry.resolve_agent(config.default_agent)
        run_settings = default_agent.run_settings
        provider = llm_registry.resolve_implementation(run_settings.llm.provider)
        context_budget = ContextBudget(
            context_window=provider.context_window(run_settings.llm.model),
            compaction_threshold=config.compaction.threshold,
        )

        # 6. conversation
        return Conversation(
            agent_registry=agent_registry,
            run_settings=run_settings,
            llm_registry=llm_registry,
            tool_registry=tool_registry,
            message_char_limit=config.message_char_limit,
            context_budget=context_budget,
            compaction_service=compaction_service,
        )

    def _build_llm_registry(self, config: ConversationConfig) -> LLMRegistry:
        registry = LLMRegistry()
        for provider_config in config.llm_registry:
            provider = self._llm_implementations.get(provider_config.provider)
            if provider is None:
                raise ConfigError(
                    f"No implementation for provider {provider_config.provider!r}. "
                    f"Available: {sorted(self._llm_implementations)}"
                )
            for model_config in provider_config.models:
                registry.register(
                    LLM(provider=provider_config.provider, model=model_config.model),
                    model_config.settings,
                    provider,
                )
        return registry

    def _build_tool_registry(self, config: ConversationConfig) -> ToolRegistry:
        registry = ToolRegistry()
        for tool_config in config.tool_registry:
            cls = self._tool_implementations.get(tool_config.type)
            if cls is None:
                raise ConfigError(
                    f"No implementation for tool type {tool_config.type!r}. "
                    f"Available: {sorted(self._tool_implementations)}"
                )
            tool = Tool(type=tool_config.type, name=tool_config.name)
            implementation = cls(tool_config)
            registry.register(tool, implementation)
        return registry

    def _build_agent_registry(
        self,
        config: ConversationConfig,
        agent_factory: AgentFactory,
    ) -> AgentRegistry:
        registry = AgentRegistry()
        for agent_config in config.agent_registry:
            agent = agent_factory.build(agent_config)
            registry.register(agent_config.name, agent)
        return registry
