from functools import reduce
from typing import TypedDict

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
    """Generate a dataframe with data about the univariate distribution properties of
    all domain classes of a field.

    Args:
        df_ranges (DataFrame): The input dataframe. See below for its expected schema.
        field (AttributeField): The field being analyzed.
        percentiles (tuple[int, ...]): Tuple containing the percentiles we wish to
            calculate for this variable.
        n_bins (int): Number of bins to use for histogram calculation.

    Returns:
        DataFrame: Output dataframe. See below for its schema.

    Schemas:
        Input: The input dataframe must have two columns. One column must be named
            'raw', and contains the original values of the field. The other column
            must be named 'domain', and contain the domain class id associated with
            that value. Only range domains should be passed to this function. It will
            still work with other types of domain, but the calculation will be moot.

        Output: The output dataframe will have a line for each domain encountered in
            the data, identified by the 'domain' column (Primary key). There will be
            a column of type float/double for each percentile value, named
            pctl_{percentile} and an array of rows field with the histogram data.
    """
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
    """This function transposes the distribution dataframe (See get_field_distribution)
    to get it into a fixed schema that is independent of the number of percentiles
    calculated.


    Args:
        df_distribution (DataFrame): The input dataframe. See the documentation for
            get_field_distribution for its schema.
        percentiles (tuple[int, ...]): The percentiles that need to be calculated.

    Returns:
        DataFrame: The transposed dataframe. See below for its schema.

    Schemas:
        Output: This dataframe will have a line for each 'attribute', 'domain' and
        'percentile', which are also the primary key of the dataframe. Each row
        represents a single percentile of a single domain class in a field. The column
        'value' contains the actual percentile value.
    """
    stack_args = [lit(len(percentiles))]
    for p in percentiles:
        stack_args.extend([lit(p), col(f"pctl_{p}")])

    return df_distribution.select(
        "attribute",
        "domain",
        stack(*stack_args).alias("percentile", "value"),
    )


def get_field_histogram(df_distribution: DataFrame) -> DataFrame:
    """This function 'explodes' the histogram field in the df_distribution, changing
    its schema into a schema of only atomic and immutable types (No arrays/objects).

    Args:
        df_distribution (DataFrame): The input dataframe. See the documentation of
            get_field_distribution for its schema.

    Returns:
        DataFrame: The 'exploded' dataframe. See below for its schema.

    Schemas:
        Output: This dataframe will have a line for each 'attribute', 'domain' and
        'bin'. There will also be a 'value' and a 'count' field. Each row represents an
        histogram bar for an attribute and domain.
    """
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
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Generates the univariate distribution metrics for a single field in a dataframe.

    Args:
        df (DataFrame): The input dataframe.
        field (AttributeField): The field we wish to calculate the metrics from.
        percentiles (tuple[int, ...], optional): Tuple containing the percentile values
        to calculate. Defaults to (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100).
        n_bins (int, optional): The number of bins to calculate the histogram. Defaults
            to 30.

    Returns:
        tuple[DataFrame, DataFrame, DataFrame]: Tuple containing the counts, histogram
            and percentile dataframes. See the documentation for get_domain_counts,
            get_field_histogram and get_field_percentiles for their schema.
    """
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


CountEntry = tuple[str, int]
DomainCounts = list[CountEntry]
Counts = dict[str, DomainCounts]


def format_counts(df: DataFrame) -> Counts:
    """Converts the data from a count dataframe into an dictionary.

    Args:
        df (DataFrame): The counts dataframe. See get_domain_count for details on
            its schema.

    Returns:
        Counts: Dictionary of the counts. This dictionary will have an entry for
            every field, which will contain a list of 'domain', 'count' tuples
            with the data.
    """
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
    """Converts the data from an histogram dataframe into an dictionary.

    Args:
        df (DataFrame): The histogram dataframe. See get_field_histogram for details on
            its schema.

    Returns:
        Histograms: Dictionary of the histograms. This dictionary will have an entry for
            everyfield, which in turn will contain a dictionary with an entry for every
            domain. This entry will consist of an 'bin', 'value', 'count' tuple with
            the data for the histogram.
    """
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
    """Converts the data from the histogram dataframe into a dictionary.

    Args:
        df (DataFrame): The percentile dataframe. See get_field_percentiles for details
            on its schema.

    Returns:
        Percentiles: Dictionary of the percentiles. It will contain and entry for each
            field, which will be another dictionary. This dictionary will contain and
            entry for each domain, which will contain a list of 'percentile', 'value'
            tuples that will contain the percentile data.
    """
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


class AttributeData(TypedDict):
    """Dictionary containing all the data generated from an Attribute Field."""

    histogram: dict[str, DomainHistogram]
    count: DomainCounts
    percentile: dict[str, DomainPercentiles]


TableData = dict[str, AttributeData]


def generate_table_metrics(df: DataFrame, table: Table) -> TableData:
    """Generates all metrics for a table.

    Args:
        df (DataFrame): The input dataset.
        table (Table): The Table metadata object.

    Returns:
        TableData: Dictionary containing all metrics for the table.
    """
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
