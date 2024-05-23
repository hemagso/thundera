""" generators.py

Contains functions that generate valid values from a Field specification. Useful when
simulating data for testing.
"""

from math import isinf
from random import choice, uniform

from .metadata import AttributeField, Domain, NullDomain, RangeDomain, SingleDomain


def sample_field(field: AttributeField, size: int) -> list[str | float | None]:
    """Sample an array of values from a field.

    Args:
        field (AttributeField): The field being sampled.
        size (int): The size of the array being generated.

    Returns:
        list[str | float | None]: The array containing the generated data.
    """
    return [sample_field_value(field) for _ in range(0, size)]


def sample_field_value(field: AttributeField) -> str | float | None:
    """Sample a single value from a field.

    This function will randomly sample (With uniform probability) a domain in a field
    and then sample a value from this domain.

    Args:
        field (AttributeField): The field being sampled.

    Returns:
        str | float | None: The generated value.
    """
    domain = choice(field.domains)
    return sample_domain(domain)


def sample_domain(domain: Domain) -> str | float | None:
    """Sample a single value from a domain.

    Args:
        domain (Domain): The domain being sampled.

    Returns:
        str | float | None: The sampled value.
    """
    if isinstance(domain, RangeDomain):
        return sample_range(domain)
    if isinstance(domain, SingleDomain):
        return domain.value
    if isinstance(domain, NullDomain):
        return None


def sample_range(domain: RangeDomain, inf_proxy: float = 1e6) -> float:
    """Sample a single value from a RangeDomain.

    Args:
        domain (RangeDomain): The range domain being sampled.
        inf_proxy (_type_, optional): Large float value that will be used as a limit
            for the sampling when the range has an +inf/-inf boundary. Defaults to 1e6.

    Returns:
        float: Sampled value.
    """
    start = domain.value.start if not isinf(domain.value.start) else -inf_proxy
    end = domain.value.end if not isinf(domain.value.end) else inf_proxy
    return uniform(start, end)
