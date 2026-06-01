"""AgentFactory: builds Agent instances from AgentConfig."""

from __future__ import annotations

from typing import Final

from ai_agent.core.factories.strategy import StrategyFactory
from ai_agent.core.models.agent import AgentConfig
from ai_agent.core.models.run import RunSettings
from ai_agent.core.services.agent import Agent
from ai_agent.core.services.run import RunService


class AgentFactory:
    """Builds Agent instances from AgentConfig."""

    def __init__(self, strategy_factory: StrategyFactory) -> None:
        self._strategy_factory: Final = strategy_factory

    def build(self, config: AgentConfig) -> Agent:
        """Build an Agent from config.

        Args:
            config: Agent configuration.

        Returns:
            A configured Agent instance.
        """
        strategy = self._strategy_factory.build(config.strategy)
        run_service = RunService(strategy)
        run_settings = RunSettings(
            agent=config.name,
            llm=config.llm,
            settings=config.settings,
            tools=config.tools,
        )
        return Agent(
            name=config.name,
            description=config.description,
            system_prompt=config.system_prompt,
            run_service=run_service,
            run_settings=run_settings,
        )
