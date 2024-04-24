from functools import reduce
from typing import TypedDict

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col,
    explode,
    histogram_numeric,
    lit,
    percentile_approx,
    row_number,
    stack,
)
from pyspark.sql.window import Window

from .metadata import AttributeField, Table
from .validators import domain_selector, is_range_domain


def get_domain_counts(df: DataFrame) -> DataFrame:
    return df.groupBy("domain").count()


def get_field_distribution(
    df: DataFrame, field: AttributeField, percentiles=(0, 20, 40, 60, 80, 100)
):
    df_domains = df.select(
        col(field.name).alias("raw"),
        domain_selector(field)(col(field.name)).alias("domain"),
    )

    df_counts = get_domain_counts(df_domains).withColumn("attribute", lit(field.name))

    df_ranges = df_domains.filter(is_range_domain(field)(col("raw")))

    df_distribution = (
        df_ranges.groupBy("domain")
        .agg(
            *[
                percentile_approx("raw", percentile / 100).alias(f"pctl_{percentile}")
                for percentile in percentiles
            ]
            + [histogram_numeric("raw", lit(30)).alias("histogram")]
        )
        .withColumn("attribute", lit(field.name))
    )

    stack_args = []
    for p in percentiles:
        stack_args.extend([lit(p), f"pctl_{p}"])

    df_percentile = df_distribution.select(
        "attribute",
        "domain",
        stack(lit(len(percentiles)), *stack_args).alias("percentile", "value"),
    )

    window_spec = Window.partitionBy("attribute", "domain").orderBy("exploded.x")
    df_histogram = (
        df_distribution.select("attribute", "domain", "histogram")
        .withColumn("exploded", explode(col("histogram")))
        .withColumn("bin", row_number().over(window_spec))
        .select(
            "attribute",
            "domain",
            "bin",
            col("exploded.x").alias("value"),
            col("exploded.y").alias("count"),
        )
    )

    return df_counts, df_histogram, df_percentile


class CountData(TypedDict):
    attribute: str
    domain: str
    count: int


def format_counts(df: DataFrame) -> list[CountData]:
    names = ("attribute", "domain", "count")
    return [dict(zip(names, tuple(row))) for row in df.select(*names).collect()]


def generate_table_metrics(df: DataFrame, table: Table):
    counts, histogram, percentiles = zip(
        *[get_field_distribution(df, attribute) for attribute in table.attributes]
    )
    df_counts = reduce(DataFrame.unionByName, counts)
    df_histogram = reduce(DataFrame.unionByName, histogram)
    df_percentiles = reduce(DataFrame.unionByName, percentiles)

    return df_counts, df_histogram, df_percentiles
