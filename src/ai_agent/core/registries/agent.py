"""Agent registry mapping names to AgentNode instances."""

from __future__ import annotations

from ai_agent.core.agent import AgentNode
from ai_agent.core.exceptions import AgentNotFoundError
from ai_agent.core.models.agent import Agent


class AgentRegistry:
    """Maps agent names to AgentNode instances."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentNode] = {}

    def register(self, agent: Agent, node: AgentNode) -> None:
        """Register an AgentNode under the given agent identity. First registration wins.

        Args:
            agent: Agent identity.
            node: Runtime AgentNode.
        """
        self._agents.setdefault(agent.name, node)

    def resolve(self, agent: Agent) -> AgentNode:
        """Return the AgentNode registered under the given agent identity.

        Args:
            agent: Agent identity.

        Raises:
            AgentNotFoundError: If no agent with that name is registered.
        """
        try:
            return self._agents[agent.name]
        except KeyError:
            raise AgentNotFoundError(
                f"No agent named {agent.name!r}. Registered: {sorted(self._agents)}"
            ) from None

    @property
    def agents(self) -> list[str]:
        """All registered agent names in registration order."""
        return list(self._agents.keys())
