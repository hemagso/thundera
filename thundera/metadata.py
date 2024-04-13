from pydantic import BaseModel, Field
from typing import Literal, Annotated
from pyspark.sql import Column
from pyspark.sql.functions import lit


def parse_range(range: str) -> tuple[float, float, bool, bool]:
    range = range.strip()

    include_start = range[0] == "["
    include_end = range[-1] == "]"

    range = range[1:-1]
    start, end = (item.strip() for item in range.split(","))
    start = float("-inf") if start == "-inf" else float(start)
    end = float("inf") if end in ("inf", "+inf") else float(end)

    return (start, end, include_start, include_end)


class RangeDomain(BaseModel):
    type: Literal["range"] = "range"
    start: float
    include_start: bool
    end: float
    include_end: bool
    description: str | None

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