from generate_data import generate
from pyspark.sql import SparkSession

from thundera.metadata import Table
from thundera.metrics import generate_table_metrics


def main():
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("local-tests")
        .config("spark.executor.cores", "1")
        .config("spark.executor.instances", "1")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )

    data = generate(spark, 1000)

    table = Table.from_yaml("./docs/examples/metadata.yml")
    df_counts, df_distribution = generate_table_metrics(data, table)
    df_counts.show()
    df_distribution.show()


if __name__ == "__main__":
    main()
