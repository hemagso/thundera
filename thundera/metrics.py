from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, histogram_numeric, lit, percentile_approx

from .metadata import AttributeField, Table
from .validators import domain_selector, is_range_domain


def get_domain_counts(df: DataFrame) -> DataFrame:
    return df.groupBy("domain").count()


def get_field_distribution(df: DataFrame, field: AttributeField):
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
                for percentile in range(0, 101, 20)
            ]
            + [histogram_numeric("raw", lit(30)).alias("histogram")]
        )
        .withColumn("attribute", lit(field.name))
    )

    return df_counts, df_distribution


def generate_table_metrics(df: DataFrame, table: Table):
    counts, distribution = zip(
        *[get_field_distribution(df, attribute) for attribute in table.attributes]
    )
    df_counts = reduce(DataFrame.unionByName, counts)
    df_distribution = reduce(DataFrame.unionByName, distribution)

    return df_counts, df_distribution
