"""Strategy identity and configuration models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Strategy(BaseModel):
    """Strategy identity."""

    model_config = ConfigDict(frozen=True)

    type: str = Field(description="Strategy type identifier.")


class StrategyConfig(Strategy):
    """Base settings for all reasoning strategy types.

    Concrete settings subclass this with a Literal 'type' field and add their own fields.
    When multiple types exist, replace this class with a discriminated union:

        StrategyConfig = Annotated[Union[FooStrategyConfig, BarStrategyConfig], Field(discriminator="type")]
    """

    max_turns: int = Field(
        default=10,
        description="Maximum reasoning turns before LoopDetectedError is raised.",
        ge=1,
    )
