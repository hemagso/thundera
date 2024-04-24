from random import choices, lognormvariate, normalvariate, uniform

from pyspark.sql import DataFrame, SparkSession


def generate_var_a(size: int) -> list[float | None]:
    kind = choices(
        [lambda: None, lambda: -99.0, lambda: -999.0, lambda: normalvariate(100, 10)],
        weights=[5, 10, 10, 75],
        k=size,
    )
    return [f() for f in kind]


def generate_var_b(size: int) -> list[float]:
    kind = choices(
        [lambda: uniform(-100, 100), lambda: -101.0, lambda: -201.0],
        weights=[30, 65, 5],
        k=size,
    )
    return [f() for f in kind]


def generate(spark: SparkSession, size: int) -> DataFrame:
    var_a = generate_var_a(size)
    var_b = generate_var_b(size)

    df = spark.createDataFrame(zip(var_a, var_b), ("var_a", "var_b"))
    return df
