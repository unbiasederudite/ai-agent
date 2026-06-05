"""Strategy identity and configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strategy(BaseModel):
    """Strategy identity."""

    model_config = ConfigDict(frozen=True)

    type: str = Field(description="Strategy type identifier.")


class BaseStrategyConfig(Strategy):
    """Base settings for all reasoning strategy types."""

    max_turns: int = Field(
        default=10,
        description="Maximum reasoning turns before LoopDetectedError is raised.",
        ge=1,
    )


class ReActStrategyConfig(BaseStrategyConfig):
    """Configuration for the ReAct (Reason + Act) reasoning strategy."""

    type: Literal["react"] = "react"


# Typed union of all concrete strategy config types. Extend when adding strategies:
#   StrategyConfig = Annotated[Union[ReActStrategyConfig, FooStrategyConfig], Field(discriminator="type")]
StrategyConfig = ReActStrategyConfig
