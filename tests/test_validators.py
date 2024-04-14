import pytest
from thundera.metadata import RangeDomain, SingleDomain, NullDomain, AttributeField
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, DoubleType
from typing import Generator, Literal
from thundera.validators import range_domain_contains, domain_contains, single_domain_contains, null_domain_contains, domain_validator


def bool_from_str(text: str) -> list[bool]:
    return [c == "1" for c in text]


def df_from_list(spark: SparkSession, values: list[float]) -> DataFrame:
    return spark.createDataFrame(
        data = [(None if v is None else float(v),) for v in values],
        schema = StructType([
            StructField("value", DoubleType(), True)
        ])
    )


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


range_domain_test_cases = [
    ("[-5, 5]", [-6, -5.0001, -5, -4.9999, 0, 4.9999, 5, 5.0001, 6, None], "0011111000"), 
    ("(-5, 5]", [-6, -5.0001, -5, -4.9999, 0, 4.9999, 5, 5.0001, 6, None], "0001111000"), 
    ("[-5, 5)", [-6, -5.0001, -5, -4.9999, 0, 4.9999, 5, 5.0001, 6, None], "0011110000"), 
    ("(-5, 5)", [-6, -5.0001, -5, -4.9999, 0, 4.9999, 5, 5.0001, 6, None], "0001110000"),
    ("(-inf, 5]", [-100_000_000, -100_000, -100, -4.9999, 0, 4.9999, 5, 5.0001, 6, None], "1111111000"), 
    ("(-inf, 5)", [-100_000_000, -100_000, -100, -4.9999, 0, 4.9999, 5, 5.0001, 6, None], "1111110000"),
    ("[-5, +inf)", [-6, -5.0001, -5, -4.9999, 0, 4.9999, 100, 100_000, 100_000_000, None], "0011111110"), 
    ("(-5, +inf)", [-6, -5.0001, -5, -4.9999, 0, 4.9999, 100, 100_000, 100_000_000, None], "0001111110") 
]

@pytest.mark.parametrize("domain_value, values, expected", range_domain_test_cases)
def test_range_domain_contains(spark: SparkSession, domain_value: str, values: list[float], expected: str):
    expected = bool_from_str(expected)
    data = df_from_list(spark, values)
    domain = RangeDomain(
        description = "Test Range",
        value = domain_value
    )

    results = data.withColumn("results", range_domain_contains(domain)(col("value")))

    assert [row.results for row in results.collect()] == expected


single_domain_test_cases = [
    (5, [-3, -2, -1, 0, 4.9999, 5, 5.0001, 6, None], "000001000")
]

@pytest.mark.parametrize("domain_value, values, expected", single_domain_test_cases)
def test_single_domain_contains(spark: SparkSession, domain_value: float, values: list[float], expected: str):
    expected = bool_from_str(expected)
    data = df_from_list(spark, values)

    domain = SingleDomain(
        description = "Test Single",
        value = domain_value
    )

    results = data.withColumn("results", single_domain_contains(domain)(col("value")))

    assert [row.results for row in results.collect()] == expected


null_domain_test_cases = [
    ([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, None], "000000000001")
]

@pytest.mark.parametrize("values, expected", null_domain_test_cases)
def test_null_domain_contains(spark: SparkSession, values: list[float], expected: str):
    expected = bool_from_str(expected)
    data = df_from_list(spark, values)

    domain = NullDomain(description="Test Null")

    results = data.withColumn("results", null_domain_contains()(col("value")))
    assert [row.results for row in results.collect()] == expected


domain_test_cases = (
    [("range",) + test_case for test_case in range_domain_test_cases] +
    [("single",) + test_case for test_case in single_domain_test_cases] +
    [("null", None) + test_case for test_case in null_domain_test_cases]
)

@pytest.mark.parametrize("domain_type, domain_value, values, expected", domain_test_cases)
def test_domain_contains(spark: SparkSession, domain_type: Literal["range", "single", "null"], domain_value, values: list[float], expected: str):
    expected = bool_from_str(expected)
    data = df_from_list(spark, values)

    if domain_type == "range":
        domain = RangeDomain(
            description= "Test Range",
            value = domain_value
        )
    elif domain_type == "single":
        domain = SingleDomain(
            description = "Test Single",
            value = domain_value
        )
    elif domain_type == "null":
        domain = NullDomain(description = "Test Null")

    results = data.withColumn("results", domain_contains(domain)(col("value")))
    
    assert [row.results for row in results.collect()] == expected

@pytest.mark.parametrize("values, expected", [
    ([-10, -5.001, -5, -4.999, -2, 0, 0.0001, 0.5, 0.999, 1, 1.0001, 49, 50, 51, 99.999, 100, 100.0001, 150, 199.9999, 200], "00100011110010011110")
])
def test_domain_validator(spark: SparkSession, values: list[float], expected: str):
    range_1 = RangeDomain(description="Range 1", value="(0, 1]")
    range_2 = RangeDomain(description="Range 2", value="[100, 200)")
    value_1 = SingleDomain(description = "Value 1", value=50)
    value_2 = SingleDomain(description = "Value 2", value=-5)
    value_3 = NullDomain(description="Null domain")

    field = AttributeField(
        name = "Test Field",
        description= "Teste Field",
        domains = [range_1, range_2, value_1, value_2, value_3]
    )

    expected = bool_from_str(expected)
    data = df_from_list(spark, values)

    results = data.withColumn("results", domain_validator(field)(col("value")))

    assert [row.results for row in results.collect()] == expected