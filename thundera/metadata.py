from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator


def parse_range(range: str) -> dict:
    range = range.strip()

    open_range = range[0]
    close_range = range[-1]
    range = range[1:-1]

    if open_range not in ("(", "["):
        raise ValueError("Invalid range format")
    if close_range not in (")", "]"):
        raise ValueError("Invalid range format")

    include_start = open_range == "["
    include_end = close_range == "]"

    start_str, end_str = (item.strip() for item in range.split(","))
    start = float("-inf") if start_str == "-inf" else float(start_str)
    end = float("inf") if end_str in ("inf", "+inf") else float(end_str)

    return {
        "start": start,
        "end": end,
        "include_start": include_start,
        "include_end": include_end,
    }


class Range(BaseModel):
    start: float
    end: float
    include_start: bool
    include_end: bool

    @model_validator(mode="before")
    @classmethod
    def parse_interval(cls, range: dict | str) -> dict:
        if isinstance(range, str):
            range = parse_range(range)
        return range

    @model_validator(mode="after")
    def check_range_bounds(self) -> Self:
        if self.start > self.end:
            raise ValueError(
                "Start value should be less than or equal to end. "
                f"Got start = {self.start} and end = {self.end}"
            )
        return self


class RangeDomain(BaseModel):
    type: Literal["range"] = "range"
    value: Range
    description: str | None


class SingleDomain(BaseModel):
    type: Literal["single"] = "single"
    value: Annotated[int | float | str, Field(union_mode="smart")]
    description: str | None


class NullDomain(BaseModel):
    type: Literal["null"] = "null"
    description: str | None


Domain = Annotated[RangeDomain | SingleDomain | NullDomain, Field(discriminator="type")]


class AttributeField(BaseModel):
    name: str
    description: str
    domains: list[Domain]


class Table(BaseModel):
    database: str
    name: str
    variables: list[AttributeField]
