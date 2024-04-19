from math import isinf
from random import choice, uniform

from .metadata import AttributeField, Domain, NullDomain, RangeDomain, SingleDomain


def sample_field(field: AttributeField, size: int) -> list[str | float | None]:
    return [sample_field_value(field) for _ in range(0, size)]


def sample_field_value(field: AttributeField) -> str | float | None:
    domain = choice(field.domains)
    return sample_domain(domain)


def sample_domain(domain: Domain) -> str | float | None:
    if isinstance(domain, RangeDomain):
        return sample_range(domain)
    if isinstance(domain, SingleDomain):
        return domain.value
    if isinstance(domain, NullDomain):
        return None


def sample_range(domain: RangeDomain, inf_proxy=1e6) -> float:
    start = domain.value.start if not isinf(domain.value.start) else -inf_proxy
    end = domain.value.end if not isinf(domain.value.end) else inf_proxy
    return uniform(start, end)
