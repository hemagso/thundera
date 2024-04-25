import json
from itertools import cycle
from typing import Generator, Literal

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, StructField, StructType

from thundera.metadata import (
    AttributeField,
    Domain,
    NullDomain,
    RangeDomain,
    SingleDomain,
)
from thundera.validators import (
    domain_contains,
    domain_validator,
    null_domain_contains,
    range_domain_contains,
    single_domain_contains,
)


def bool_from_str(text: str) -> list[bool]:
    return [c == "1" for c in text]


def df_from_list(spark: SparkSession, values: list[float]) -> DataFrame:
    return spark.createDataFrame(
        data=[(None if v is None else float(v),) for v in values],
        schema=StructType([StructField("value", DoubleType(), True)]),
    )


def parametrize_from_file(filename: str):
    def decorator(func):
        with open(filename, "r", encoding="utf-8") as file:
            specs = json.load(file)
        args = {}
        for arg in specs["argnames"]:
            args[arg] = [item["args"][arg] for item in specs["expected"]]

        expected = [bool_from_str(item["expected"]) for item in specs["expected"]]

        argnames = ", ".join(specs["argnames"]) + ", values, expected"
        argvalues = list(zip(*args.values(), cycle([specs["data"]]), expected))

        return pytest.mark.parametrize(argnames=argnames, argvalues=argvalues)(func)

    return decorator


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("local-tests")
        .config("spark.executor.cores", "1")
        .config("spark.executor.instances", "1")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@parametrize_from_file("./tests/fixtures/data/range_domain.json")
def test_range_domain_contains(
    spark: SparkSession, domain_value: str, values: list[float], expected: list[bool]
):
    data = df_from_list(spark, values)
    domain = RangeDomain(description="Test Range", value=domain_value, id="range")

    results = data.withColumn("results", range_domain_contains(domain)(col("value")))

    assert [row.results for row in results.collect()] == expected


single_domain_test_cases = [
    (5, [-3, -2, -1, 0, 4.9999, 5, 5.0001, 6, None], "000001000")
]


@parametrize_from_file("./tests/fixtures/data/single_domain.json")
def test_single_domain_contains(
    spark: SparkSession, domain_value: float, values: list[float], expected: str
):
    data = df_from_list(spark, values)

    domain = SingleDomain(description="Test Single", value=domain_value, id="single")

    results = data.withColumn("results", single_domain_contains(domain)(col("value")))

    assert [row.results for row in results.collect()] == expected


null_domain_test_cases = [
    ([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, None], "000000000001")
]


@parametrize_from_file("./tests/fixtures/data/null_domain.json")
def test_null_domain_contains(spark: SparkSession, values: list[float], expected: str):
    data = df_from_list(spark, values)

    results = data.withColumn("results", null_domain_contains()(col("value")))
    assert [row.results for row in results.collect()] == expected


@parametrize_from_file("./tests/fixtures/data/domain_contains.json")
def test_domain_contains(
    spark: SparkSession,
    domain_type: Literal["range", "single", "null"],
    domain_value,
    values: list[float],
    expected: str,
):
    data = df_from_list(spark, values)

    domain: Domain
    if domain_type == "range":
        domain = RangeDomain(description="Test Range", value=domain_value, id="range")
    elif domain_type == "single":
        domain = SingleDomain(
            description="Test Single", value=domain_value, id="single"
        )
    elif domain_type == "null":
        domain = NullDomain(description="Test Null", id="null")

    results = data.withColumn("results", domain_contains(domain)(col("value")))

    assert [row.results for row in results.collect()] == expected


@pytest.mark.parametrize(
    "values, expected",
    [
        (
            [
                -10,
                -5.001,
                -5,
                -4.999,
                -2,
                0,
                0.0001,
                0.5,
                0.999,
                1,
                1.0001,
                49,
                50,
                51,
                99.999,
                100,
                100.0001,
                150,
                199.9999,
                200,
            ],
            "00100011110010011110",
        )
    ],
)
def test_domain_validator(spark: SparkSession, values: list[float], expected: str):
    range_1 = RangeDomain(description="Range 1", value="(0, 1]", id="range_1")
    range_2 = RangeDomain(description="Range 2", value="[100, 200)", id="range_2")
    value_1 = SingleDomain(description="Value 1", value=50, id="value_1")
    value_2 = SingleDomain(description="Value 2", value=-5, id="value_2")
    value_3 = NullDomain(description="Null domain", id="value_3")

    field = AttributeField(
        name="Test Field",
        description="Test Field",
        domains=[range_1, range_2, value_1, value_2, value_3],
    )

    expected_bool = bool_from_str(expected)
    data = df_from_list(spark, values)

    results = data.withColumn("results", domain_validator(field)(col("value")))

    assert [row.results for row in results.collect()] == expected_bool


@pytest.mark.parametrize(
    "expected, message", [(ValueError, "Can't validate a field with no valid domains")]
)
def test_fail_domain_validator(expected: type[Exception], message: str):
    field = AttributeField(name="Test Field", description="Test Field", domains=[])
    with pytest.raises(expected, match=message):
        domain_validator(field)
