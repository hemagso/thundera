import pytest
from thundera.metadata import RangeDomain, parse_range
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, DoubleType, BooleanType
from typing import Generator


def bool_from_str(text: str) -> list[bool]:
    return [c == "1" for c in text]


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


@pytest.mark.parametrize("range, values, expected", [
    ("[-5, 5]", [-6, -5.0001, -5, -4.9999, 0, 4.9999, 5, 5.0001, 6, None], "0011111000"), 
    ("(-5, 5]", [-6, -5.0001, -5, -4.9999, 0, 4.9999, 5, 5.0001, 6, None], "0001111000"), 
    ("[-5, 5)", [-6, -5.0001, -5, -4.9999, 0, 4.9999, 5, 5.0001, 6, None], "0011110000"), 
    ("(-5, 5)", [-6, -5.0001, -5, -4.9999, 0, 4.9999, 5, 5.0001, 6, None], "0001110000"),
    ("(-inf, 5]", [-100_000_000, -100_000, -100, -4.9999, 0, 4.9999, 5, 5.0001, 6, None], "1111111000"), 
    ("(-inf, 5)", [-100_000_000, -100_000, -100, -4.9999, 0, 4.9999, 5, 5.0001, 6, None], "1111110000"),
    ("[-5, +inf)", [-6, -5.0001, -5, -4.9999, 0, 4.9999, 100, 100_000, 100_000_000, None], "0011111110"), 
    ("(-5, +inf)", [-6, -5.0001, -5, -4.9999, 0, 4.9999, 100, 100_000, 100_000_000, None], "0001111110") 
])
def test_range_domain_contains(spark: SparkSession, range: str, values: list[float], expected: list[bool]):
    expected = bool_from_str(expected)
    data = spark.createDataFrame(
        data = [(None if v is None else float(v),) for v in values],
        schema = StructType([
            StructField("value", DoubleType(), True)
        ])
    )

    domain = RangeDomain(
        description = "Test Range",
        value = range
    )

    results = data.withColumn("results", domain.contains(col("value")))

    assert [row.results for row in results.collect()] == expected


@pytest.mark.parametrize("range, expected", [
    ("[-5, 5]", (-5, 5, True, True)),
    ("(-5, 5]", (-5, 5, False, True)),
    ("[-5, 5)", (-5, 5, True, False)),
    ("(-5, 5)", (-5, 5, False, False))
])
def test_parse_range(range: str, expected: tuple[float, float, bool, bool]):
    assert parse_range(range) == dict(zip(("start", "end", "include_start", "include_end"), expected))