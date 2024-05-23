from typing import Annotated, Literal, Self, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator
from yaml import safe_load


class RangeDict(TypedDict):
    start: float | int
    end: float | int
    include_start: bool
    include_end: bool


def parse_range(range: str) -> RangeDict:
    """Parse a range specification string into a RangeDomain compatible dictionary. The
    string is expected to follow ISO 31-11 set notation, where a parentheses indicates
    an open boundary (That is, the boundary is not included in the range) and a square
    bracket indicates a closed boundary (That is, the boundary is included in the range)

    Args:
        range (str): String describing a range using ISO 31-11 interval notation.

    Raises:
        ValueError: Raised if the input string does not conforms to the ISO 31-11
            interval notation.

    Returns:
        RangeDict: Dictionary representing the range.
    """
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
    """This class represents a range, or mathematical interval.

    Attributes:
        start (float): The start boundary of the range.
        end (float): The end boundary of the range.
        include_start (bool): Whether the start boundary value is included in the range
            itself.
        include_end (bool): Whether the end boundary value is included in the range
            itself
    """

    start: float
    end: float
    include_start: bool
    include_end: bool

    @model_validator(mode="after")
    def check_range_bounds(self) -> Self:
        """Validates whether the start boundary is less then or equal to the end
        boundary

        Raises:
            ValueError: Raised if the start boundary is greater than the end boundary.

        Returns:
            Self: Returns the validated instance.
        """
        if self.start > self.end:
            raise ValueError(
                "Start value should be less than or equal to end. "
                f"Got start = {self.start} and end = {self.end}"
            )
        return self


class RangeDomain(BaseModel):
    """Represents a Domain of type Range. This domain represents an interval of valid
    values.

    Attributes:
        type (Literal["range"]): A literal indicating the type of the domain. Used as
            a field discriminator by pydantic when parsing dictionaries into models.
        id (str):"String that uniquely identifies this domain within the valid domains
            of a field.
        value (Range): A Range object describing the valid interval for this domain.
        description (str): A string that describes what values in this range represent.
    """

    type: Literal["range"] = "range"
    id: str
    value: Range
    description: str | None

    @field_validator("value", mode="before")
    @classmethod
    def parse_range(cls, range: str | RangeDict | Range) -> Range:
        """This methods converts the range attribute from a string that represents a
        range (Any string accepted by the parse_range method) or from a RangeDict
        dictionary.

        Args:
            range (str | RangeDict | Range): A valid range representation.

        Returns:
            Range: A range object representing the range.
        """
        if isinstance(range, Range):
            return range
        if isinstance(range, str):
            range = parse_range(range)
        return Range(**range)


class SingleDomain(BaseModel):
    """Represents a Domain of type Single. This domain contains a single value.

    Attributes:
        type (Literal["single"]): A literal indicating the type of the domain. Used as
            a field discriminator by pydantic when parsing dictionaries into models.
        id (str):"String that uniquely identifies this domain within the valid domains
            of a field.
        value (int | float | str): The value represented by this domain.
        description (str): A string that describes what values in this range represent.
    """

    type: Literal["single"] = "single"
    id: str
    value: Annotated[int | float | str, Field(union_mode="smart")]
    description: str | None


class NullDomain(BaseModel):
    """Represents a Domain of type Null. This domain represents the absence of values.

    Attributes:
        type (Literal["null"]): A literal indicating the type of the domain. Used as
            a field discriminator by pydantic when parsing dictionaries into models.
        id (str):"String that uniquely identifies this domain within the valid domains
            of a field.
        description (str): A string that describes what values in this range represent.
    """

    id: str
    type: Literal["null"] = "null"
    description: str | None


Domain = Annotated[RangeDomain | SingleDomain | NullDomain, Field(discriminator="type")]


class AttributeField(BaseModel):
    """Represents an Attribute field in a table. An Attribute Field is a Field that
    contains a non-key attribute of the entity represented in a table.

    Attributes:
        name (str): Name of the field. Must correspond to the physical name of a column
            in the parent table.
        description (str): Informative description of a field.
        domains (list[Domain]): List of valid domains for a field.
    """

    name: str
    description: str
    domains: list[Domain]


class Table(BaseModel):
    """Represents a table containing data about some entity.

    Attributes:
        database (str): The name of the database where the table is located.
        name (str): The name of the table.
        attributes (list[AttributeField]): List of attribute fields present in the
            table.
    """

    database: str
    name: str
    attributes: list[AttributeField]

    @classmethod
    def from_yaml(cls: type[Self], path: str) -> Self:
        with open(path, "r", encoding="utf-8") as file:
            d = safe_load(file)
            return cls(**d)
