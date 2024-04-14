from typing import Callable

from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import lit

from .metadata import AttributeField, Domain, RangeDomain, SingleDomain

DomainValidator = Callable[[Column], Column]

def domain_contains(domain: Domain) -> DomainValidator:
    if domain.type == "range":
        return range_domain_contains(domain)
    if domain.type == "single":
        return single_domain_contains(domain)
    if domain.type == "null":
        return null_domain_contains()
    
def range_domain_contains(domain: RangeDomain) -> DomainValidator:
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
    def validator(values: Column) -> Column:
        return values.isNotNull() & (values == lit(domain.value))
    return validator

def null_domain_contains() -> DomainValidator:
    def validator(values: Column) -> Column:
        return values.isNull()
    return validator


def domain_validator(field: AttributeField) -> DomainValidator:
    def validator(values: Column) -> Column:
        expression = None
        for domain in field.domains:
            if expression is None:
                expression = domain_contains(domain)(values)
            else:
                expression |= domain_contains(domain)(values)
        return expression
    return validator