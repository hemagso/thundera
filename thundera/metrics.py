from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col,
    collect_list,
    explode,
    histogram_numeric,
    lit,
    percentile_approx,
    row_number,
    stack,
    struct,
)
from pyspark.sql.window import Window

from .metadata import AttributeField, Table
from .validators import domain_selector, is_range_domain


def get_domain_counts(df: DataFrame) -> DataFrame:
    """Generate a dataset with the count of each domain

    TODO: Merge back with the list of valid domains to detect domains with no counts

    Args:
        df (DataFrame): Dataframe with a 'domain' field already calculated identifying
            each domain.

    Returns:
        DataFrame: Dataframe with the domain counts.
    """
    return df.groupBy("domain").count()


def get_field_distribution(
    df_ranges: DataFrame,
    field: AttributeField,
    percentiles: tuple[int, ...],
    n_bins: int,
) -> DataFrame:
    return (
        df_ranges.groupBy("domain")
        .agg(
            *[
                percentile_approx("raw", percentile / 100).alias(f"pctl_{percentile}")
                for percentile in percentiles
            ]
            + [histogram_numeric("raw", lit(n_bins)).alias("histogram")]
        )
        .withColumn("attribute", lit(field.name))
    )


def get_field_percentiles(
    df_distribution: DataFrame, percentiles: tuple[int, ...]
) -> DataFrame:
    stack_args = [lit(len(percentiles))]
    for p in percentiles:
        stack_args.extend([lit(p), col(f"pctl_{p}")])

    return df_distribution.select(
        "attribute",
        "domain",
        stack(*stack_args).alias("percentile", "value"),
    )


def get_field_histogram(df_distribution: DataFrame) -> DataFrame:
    window_spec = Window.partitionBy("attribute", "domain").orderBy("exploded.x")
    return (
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


def get_field_metrics(
    df: DataFrame,
    field: AttributeField,
    percentiles: tuple[int, ...] = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100),
    n_bins: int = 30,
):
    df_domains = df.select(
        col(field.name).alias("raw"),
        domain_selector(field)(col(field.name)).alias("domain"),
    )

    df_counts = get_domain_counts(df_domains).withColumn("attribute", lit(field.name))
    df_ranges = df_domains.filter(is_range_domain(field)(col("raw")))
    df_distribution = get_field_distribution(df_ranges, field, percentiles, n_bins)
    df_percentile = get_field_percentiles(df_distribution, percentiles)
    df_histogram = get_field_histogram(df_distribution)

    return df_counts, df_histogram, df_percentile


def format_counts(df: DataFrame):
    grouped_df = df.groupBy("attribute").agg(
        collect_list(struct("domain", "count")).alias("domain_counts")
    )

    return {
        row["attribute"]: [(r["domain"], r["count"]) for r in row["domain_counts"]]
        for row in grouped_df.collect()
    }


HistogramEntry = tuple[int, float, int]
DomainHistogram = list[HistogramEntry]
Histograms = dict[str, dict[str, DomainHistogram]]


def format_histograms(df: DataFrame) -> Histograms:
    grouped_df = df.groupBy("attribute", "domain").agg(
        collect_list(struct("bin", "value", "count")).alias("bin_counts")
    )
    result: Histograms = {}
    for row in grouped_df.collect():
        attribute = row["attribute"]
        domain = row["domain"]
        counts = [(r["bin"], r["value"], r["count"]) for r in row["bin_counts"]]

        if attribute not in result:
            result[attribute] = {}

        result[attribute][domain] = counts

    return result


PercentileEntry = tuple[int, float]
DomainPercentiles = list[PercentileEntry]
Percentiles = dict[str, dict[str, DomainPercentiles]]


def format_percentiles(df: DataFrame) -> Percentiles:
    grouped_df = df.groupBy("attribute", "domain").agg(
        collect_list(struct("percentile", "value")).alias("pctl_values")
    )
    result: Percentiles = {}
    for row in grouped_df.collect():
        attribute = row["attribute"]
        domain = row["domain"]
        counts = [(r["percentile"], r["value"]) for r in row["pctl_values"]]

        if attribute not in result:
            result[attribute] = {}

        result[attribute][domain] = counts

    return result


def generate_table_metrics(df: DataFrame, table: Table):
    counts, histogram, percentiles = zip(
        *[get_field_metrics(df, attribute) for attribute in table.attributes]
    )
    df_counts = reduce(DataFrame.unionByName, counts)
    df_histogram = reduce(DataFrame.unionByName, histogram)
    df_percentiles = reduce(DataFrame.unionByName, percentiles)

    data_counts = format_counts(df_counts)
    data_histograms = format_histograms(df_histogram)
    data_percentiles = format_percentiles(df_percentiles)

    return {
        attribute.name: {
            "histogram": data_histograms[attribute.name],
            "count": data_counts[attribute.name],
            "percentile": data_percentiles[attribute.name],
        }
        for attribute in table.attributes
    }
