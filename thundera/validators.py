from functools import reduce
from typing import Callable

from pyspark.sql import Column
from pyspark.sql.functions import lit, when

from .metadata import AttributeField, Domain, RangeDomain, SingleDomain

DomainValidator = Callable[[Column], Column]
DomainTransformer = Callable[[Column], Column]


def domain_contains(domain: Domain) -> DomainValidator:
    """Returns a function that builds a spark expression that validates whether values
    in a spark Column are within a range domain.

    Args:
        domain (Domain): The domain being checked against.

    Returns:
        DomainValidator: A function that returns the spark expression that performs
            the validation.
    """
    if domain.type == "range":
        return range_domain_contains(domain)
    if domain.type == "single":
        return single_domain_contains(domain)
    if domain.type == "null":
        return null_domain_contains()


def range_domain_contains(domain: RangeDomain) -> DomainValidator:
    """Returns a function that builds a spark expression that validates whether values
    in a Spark Column are within a range domain.

    Args:
        domain (RangeDomain): The range domain being checked against.

    Returns:
        DomainValidator: A function that returns the spark expression that performs
            the validation.
    """

    def validator(values: Column) -> Column:
        range = domain.value
        start = lit(range.start)
        end = lit(range.end)

        if range.include_start and range.include_end:
            return values.isNotNull() & (values >= start) & (values <= end)
        if range.include_start:
            return values.isNotNull() & (values >= start) & (values < end)
        if range.include_end:
            return values.isNotNull() & (values > start) & (values <= end)
        return values.isNotNull() & (values > start) & (values < end)

    return validator


def single_domain_contains(domain: SingleDomain) -> DomainValidator:
    """Returns a function that builds a spark expression that validates whether values
    in a Spark Column are within a single domain.

    Args:
        domain (SingleDomain): The single domain being checked against.

    Returns:
        DomainValidator: A function that returns the spark expression that performs
            the validation.
    """

    def validator(values: Column) -> Column:
        return values.isNotNull() & (values == lit(domain.value))

    return validator


def null_domain_contains() -> DomainValidator:
    """Returns a function that builds a spark expression that validates whether values
    in a Spark Column are within a null domain.

    Returns:
        DomainValidator: A function that returns the spark expression that performs
            the validation.
    """

    def validator(values: Column) -> Column:
        return values.isNull()

    return validator


def domain_validator(field: AttributeField) -> DomainValidator:
    """Returns a function that builds a spark expression that validates whether values
    in a Spark Column are within an Attribute Field domain.

    Args:
        field (AttributeField): The Attribute Field being checked against.

    Raises:
        ValueError: Raised if the Attribute Field being checked against does not have
            any valid domains.

    Returns:
        DomainValidator: A function that returns the spark expression that performs
            the validation.
    """
    if len(field.domains) == 0:
        raise ValueError("Can't validate a field with no valid domains")

    def validator(values: Column) -> Column:
        return reduce(
            lambda a, b: a | b,
            [domain_contains(domain)(values) for domain in field.domains],
        )

    return validator


def domain_selector(field: AttributeField) -> DomainTransformer:
    """Returns a function that builds a spark expression that calculated which
    sub-domain of an Attribute Field each value of a Column belongs to.

    Args:
        field (AttributeField): The Attribute Field being processed against.

    Returns:
        DomainTransformer: A function that returns the spark expression that transforms
            the column.
    """

    if len(field.domains) == 0:
        raise ValueError("Can't select domains from a field with no valid domains")

    def transformer(values: Column) -> Column:
        domain = field.domains[0]
        results = when(domain_contains(domain)(values), lit(domain.description))
        for domain in field.domains[1:]:
            results = results.when(
                domain_contains(domain)(values), lit(domain.description)
            )
        results = results.otherwise(lit("__INVALID__"))
        return results

    return transformer
