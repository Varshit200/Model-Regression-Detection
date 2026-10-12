
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClassificationResult(BaseModel):
    """Validated output from the support email classifier."""

    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "billing",
        "technical",
        "account",
        "general",
    ]

    summary: str = Field(
        min_length=1,
        max_length=300,
        description="A concise, factual summary of the email.",
    )
