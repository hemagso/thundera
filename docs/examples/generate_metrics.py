from generate_data import generate
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lit, row_number, stack
from pyspark.sql.window import Window

from thundera.metadata import Table
from thundera.metrics import generate_table_metrics


def get_spark() -> SparkSession:
    return (
        SparkSession.builder.master("local[1]")
        .appName("local-tests")
        .config("spark.executor.cores", "1")
        .config("spark.executor.instances", "1")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )


def main():
    spark = get_spark()
    data = generate(spark, 10_000)

    table = Table.from_yaml("./docs/examples/metadata.yml")
    df_counts, df_histogram, df_percentile = generate_table_metrics(data, table)

    df_counts.show()
    df_histogram.show()
    df_percentile.show()


if __name__ == "__main__":
    main()
