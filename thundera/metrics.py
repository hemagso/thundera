from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, histogram_numeric, lit, percentile_approx

from .metadata import AttributeField
from .validators import domain_selector, is_range_domain


def get_domain_counts(df: DataFrame) -> DataFrame:
    return df.groupBy("domain").count()


def get_field_distribution(df: DataFrame, field: AttributeField):
    df_domains = df.select(
        col(field.name).alias("raw"),
        domain_selector(field)(col(field.name)).alias("domain"),
    )

    df_counts = get_domain_counts(df_domains)

    df_ranges = df_domains.filter(is_range_domain(field)(col("raw")))

    df_distribution = df_ranges.groupBy("domain").agg(
        *[
            percentile_approx("raw", percentile / 100).alias(f"pctl_{percentile}")
            for percentile in range(0, 101, 20)
        ]
        + [histogram_numeric("raw", lit(30)).alias("histogram")]
    )

    return df_counts, df_distribution
