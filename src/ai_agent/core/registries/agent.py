"""Agent registry mapping names to Agent instances."""

from __future__ import annotations

from ai_agent.core.exceptions import AgentNotFoundError
from ai_agent.core.services.agent import Agent


class AgentRegistry:
    """Maps agent names to Agent instances."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, name: str, agent: Agent) -> None:
        """Register an Agent under the given name. First registration wins.

        Args:
            name: Agent name.
            agent: Agent instance.
        """
        self._agents.setdefault(name, agent)

    def resolve_agent(self, name: str) -> Agent:
        """Return the Agent registered under the given name.

        Args:
            name: Agent name.

        Raises:
            AgentNotFoundError: No agent registered under that name.
        """
        try:
            return self._agents[name]
        except KeyError:
            raise AgentNotFoundError(
                f"No agent named {name!r}. Registered: {sorted(self._agents)}"
            ) from None

    @property
    def agents(self) -> list[str]:
        """All registered agent names in registration order."""
        return list(self._agents)
