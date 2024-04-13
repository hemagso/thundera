from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Literal, Annotated
from pyspark.sql import Column
from pyspark.sql.functions import lit
from typing import Any, Self

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

    start, end = (item.strip() for item in range.split(","))
    start = float("-inf") if start == "-inf" else float(start)
    end = float("inf") if end in ("inf", "+inf") else float(end)

    return {
        "start": start, 
        "end": end, 
        "include_start": include_start, 
        "include_end": include_end
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
            raise ValueError("Start value should be less of equal to end")
        return self

    def contains(self, values: Column) -> Column:
        start = lit(self.start)
        end = lit(self.end)
        if self.include_start and self.include_end:
            return values.isNotNull() & (values >= start) & (values <= end)
        if self.include_start:
            return values.isNotNull() & (values >= start) & (values < end)
        if self.include_end:
            return values.isNotNull() & (values > start) & (values <= end)
        return values.isNotNull() & (values > start) & (values < end)

class RangeDomain(BaseModel):
    type: Literal["range"] = "range"
    value: Range
    description: str | None

    def contains(self, values: Column) -> Column:
        return self.value.contains(values)
    

class SingleDomain(BaseModel):
    type: Literal["single"] = "single"
    value: Annotated[int | float | str, Field(union_mode="smart")]
    description: str | None


class NullDomain(BaseModel):
    type: Literal["null"] = "null"
    description: str | None


Domain = Annotated[RangeDomain, SingleDomain, NullDomain, Field(discriminator="type")]


class Variable(BaseModel):
    name: str
    description: str
    domains:  list[Domain]


class Table(BaseModel):
    database: str
    name: str
    variables: list[Variable]