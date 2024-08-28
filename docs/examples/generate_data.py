from random import choices, lognormvariate, normalvariate, uniform

from pyspark.sql import DataFrame, SparkSession


def generate_var_a(size: int, mu: float, sigma: float) -> list[float | None]:
    kind = choices(
        [lambda: None, lambda: -99.0, lambda: -999.0, lambda: normalvariate(mu, sigma)],
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


def generate_var_c(size: int) -> list[float | None]:
    kind = choices(
        [lambda: lognormvariate(3, 0.5), lambda: -8.0, lambda: -9.0, lambda: None],
        weights=[80, 15, 4, 1],
        k=size,
    )
    return [f() for f in kind]


def generate_var_d(size: int, mu: float, sigma: float) -> list[float | None]:
    kind = choices(
        [lambda: None, lambda: -99.0, lambda: -999.0, lambda: normalvariate(mu, sigma)],
        weights=[10, 10, 65, 15],
        k=size,
    )
    return [f() for f in kind]


def generate(spark: SparkSession, size: int, mu: float, sigma: float) -> DataFrame:
    var_a = generate_var_a(size, mu, sigma)
    var_b = generate_var_b(size)
    var_c = generate_var_c(size)
    var_d = generate_var_d(size, mu, sigma)

    df = spark.createDataFrame(
        zip(var_a, var_b, var_c, var_d), ("var_a", "var_b", "var_c", "var_d")
    )
    return df
