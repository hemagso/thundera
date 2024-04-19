from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when

from thundera.generators import sample_field
from thundera.metadata import AttributeField, NullDomain, RangeDomain, SingleDomain
from thundera.metrics import get_domain_counts, get_field_distribution
from thundera.validators import domain_contains, domain_selector

negatives = RangeDomain(value="(-10, 0)", description="Negatives")
zero = SingleDomain(value=0.0, description="Zero")
positives = RangeDomain(value="(0, 10)", description="Positives")
null = NullDomain(description="Missing")

field = AttributeField(
    name="value", description="test", domains=[negatives, zero, positives, null]
)

spark = SparkSession.builder.appName("Test").getOrCreate()

data = list(zip(sample_field(field, 10_000)))
df = spark.createDataFrame(data, ["value"])

df = df.withColumn("description", domain_selector(field)(col("value")))

df_counts, df_percentile = get_field_distribution(df, field)

df_counts.show()
df_percentile.show()

print(df_percentile.collect()[0].histogram)
