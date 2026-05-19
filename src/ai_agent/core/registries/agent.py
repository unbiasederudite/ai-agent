"""Agent registry mapping Agent identities to IAgent implementations."""

from __future__ import annotations

from ai_agent.core.exceptions import AgentNotFoundError
from ai_agent.core.models.agent import Agent
from ai_agent.core.protocols.agent import IAgent


class AgentRegistry:
    """Maps Agent identities to IAgent implementations."""

    def __init__(self) -> None:
        self._agents: dict[Agent, IAgent] = {}

    def register(self, agent: Agent, implementation: IAgent) -> None:
        """Register an IAgent under the given agent identity. First registration wins.

        Args:
            agent: Agent identity.
            implementation: Agent implementation.
        """
        self._agents.setdefault(agent, implementation)

    def resolve_implementation(self, agent: Agent) -> IAgent:
        """Return the IAgent registered under the given agent identity.

        Args:
            agent: Agent identity.

        Raises:
            AgentNotFoundError: Agent not registered.
        """
        try:
            return self._agents[agent]
        except KeyError:
            registered = [(a.type, a.name) for a in self._agents]
            raise AgentNotFoundError(
                f"No agent type={agent.type!r} name={agent.name!r}. Registered: {registered}"
            ) from None

    @property
    def agents(self) -> list[Agent]:
        """All registered agents in registration order."""
        return list(self._agents)
