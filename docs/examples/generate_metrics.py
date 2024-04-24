import json
from pprint import pprint

from generate_data import generate
from pyspark.sql import SparkSession

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
    metric_data = generate_table_metrics(data, table)

    pprint(metric_data, compact=True)
    with open("output.json", "w", encoding="utf-8") as file:
        json.dump(metric_data, file)


if __name__ == "__main__":
    main()
