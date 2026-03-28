
# PySpark Data Engineering: 30 Practical Q&A (Notebook-Style Markdown)

> **Purpose**: This notebook-style Markdown file consolidates core-to-advanced PySpark patterns you’ll use across Bronze/Silver/Gold layers, including DataFrame ops, windowed aggregations, joins, Delta Lake, and streaming. Each item includes a **concise solution** and the **reasoning behind it** so you can quickly recall the *why* not just the *how*.

---

## Part 1: Core DataFrame Operations & Data Cleaning
These tasks focus on reading, cleaning, and shaping raw data efficiently.

### 1) Read a CSV from S3 and infer schema
```python
# Assumes cluster has S3 access configured (IAM role/keys), and the path exists
# inferschema triggers a pass over the file to detect types

df = spark.read.csv("s3://bucket/path/data.csv", header=True, inferSchema=True)
```
**Explanation**: `inferSchema=True` detects column types from a sample/read pass; using `header=True` treats the first row as column names. This is simple but can be slower than specifying a schema—prefer explicit schemas in production for performance and type safety.

---

### 2) Drop rows where specific critical columns are null
```python
df_clean = df.dropna(subset=["transaction_id", "customer_id"])  # keep only rows where both are non-null
```
**Explanation**: `dropna(subset=...)` removes records missing critical business keys to ensure downstream joins and deduplication are reliable.

---

### 3) Fill null values with defaults across multiple columns
```python
df_filled = df.fillna({"status": "unknown", "amount": 0.0})
```
**Explanation**: `fillna` applies per-column default values, preventing null-propagation during computations (e.g., aggregates) and enabling safe filtering.

---

### 4) Rename multiple columns dynamically
```python
# Example: normalize spaces to underscores for SQL-friendliness
new_cols = [c.replace(" ", "_") for c in df.columns]
df_renamed = df.toDF(*new_cols)
```
**Explanation**: Renaming systematically enforces naming conventions that simplify downstream SQL queries and tooling compatibility.

---

### 5) Cast a string column to timestamp and filter last 30 days
```python
from pyspark.sql.functions import col, to_timestamp, current_date, date_sub

df_time = (
    df.withColumn("event_time", to_timestamp(col("time_str"), "yyyy-MM-dd HH:mm:ss"))
      .filter(col("event_time") >= date_sub(current_date(), 30))
)
```
**Explanation**: `to_timestamp` parses strings to `TimestampType`. Comparing with `current_date()` and `date_sub` is efficient and pushdown-friendly when possible.

---

### 6) Extract Year and Month from a date column
```python
from pyspark.sql.functions import year, month

df_ym = df.withColumn("year", year("event_date")).withColumn("month", month("event_date"))
```
**Explanation**: Adding calendar attributes supports partitioning, rollups, and dimensional modeling.

---

### 7) Flatten a nested Struct column
```python
# If `user_info` is a struct (e.g., {name: ..., age: ...})
df_flat = df.select("id", "user_info.*")
```
**Explanation**: Using `struct.*` expands nested fields into top-level columns, making them easier to query and transform.

---

## Part 2: Complex Transformations & Aggregations
Window functions and complex aggregations are staples of Silver/Gold processing.

### 8) Total, average, and max by category
```python
from pyspark.sql.functions import sum, avg, max

df_agg = df.groupBy("category").agg(
    sum("amount").alias("total_sales"),
    avg("amount").alias("avg_sales"),
    max("amount").alias("max_sale"),
)
```
**Explanation**: Grouped aggregations compute multiple metrics in a single shuffle, reducing passes over the data.

---

### 9) Efficiently count distinct in a massive dataset
```python
from pyspark.sql.functions import approx_count_distinct

# Much faster than exact countDistinct for large data
df_distinct = df.agg(approx_count_distinct("customer_id").alias("approx_unique_customers"))
```
**Explanation**: `approx_count_distinct` uses HyperLogLog++, providing near-accurate cardinality with sublinear memory and compute—ideal for big data.

---

### 10) Pivot rows to columns
```python
# Creates one row per store_id and a column per product_category, summing revenue

df_pivot = df.groupBy("store_id").pivot("product_category").sum("revenue")
```
**Explanation**: Pivoting is a shuffle-heavy operation; ensure the pivot column has bounded cardinality and consider `spark.sql.pivotMaxValues` limits.

---

### 11) Top 3 highest-paid employees per department
```python
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col

window_spec = Window.partitionBy("department").orderBy(col("salary").desc())
df_top = df.withColumn("rn", row_number().over(window_spec)).filter(col("rn") <= 3)
```
**Explanation**: Window `row_number()` ranks rows within partitions; filtering by rank yields top-k per group without a post-aggregation join.

---

### 12) Running total of sales over time
```python
from pyspark.sql.functions import sum
from pyspark.sql.window import Window

window_spec = (
    Window.partitionBy("store_id").orderBy("date").rowsBetween(Window.unboundedPreceding, Window.currentRow)
)

df_running = df.withColumn("running_total", sum("sales").over(window_spec))
```
**Explanation**: Bounded/unbounded window frames enable cumulative metrics aligned to business ordering (e.g., by date).

---

### 13) 7-day moving average
```python
from pyspark.sql.functions import avg
from pyspark.sql.window import Window

window_spec = Window.partitionBy("stock_ticker").orderBy("date").rowsBetween(-6, 0)
df_moving_avg = df.withColumn("7_day_avg", avg("price").over(window_spec))
```
**Explanation**: A sliding window over the last 7 rows (not calendar days) computes a moving average. For calendar-based windows, consider time-range windowing on timestamp columns.

---

### 14) Explode an array column into rows
```python
from pyspark.sql.functions import explode, col

df_exploded = df.withColumn("item", explode(col("items_array")))
```
**Explanation**: `explode` normalizes arrays into 1:N row-level records for downstream joins and aggregations.

---

## Part 3: Joins & Optimization
Handling joins efficiently is critical for performance and cost.

### 15) Join two DataFrames and avoid duplicate column errors
```python
# If join key has the same name in both, passing the string uses it for equality join

df_joined = df1.join(df2, "customer_id", "inner")
```
**Explanation**: Supplying the key name avoids ambiguous column specs and generates a natural equality join on that column.

---

### 16) Records in A that are NOT in B
```python
df_missing = df_A.join(df_B, "id", "left_anti")
```
**Explanation**: `left_anti` returns rows from the left DataFrame with no matches on the right—perfect for anti-semi set operations.

---

### 17) Optimize a join between a massive table and a very small table
```python
from pyspark.sql.functions import broadcast

df_joined = large_df.join(broadcast(small_df), "key")
```
**Explanation**: Broadcast joins send the small table to all executors, avoiding a shuffle of the large table and drastically cutting join time.

---

### 18) Handle data skew during a join (salting)
```python
from pyspark.sql.functions import rand, explode, array, lit

# Add salt to the skewed key to spread hot keys across partitions
num_salts = 10

df_large_salted = large_df.withColumn("salt", (rand() * num_salts).cast("int"))
df_small_exploded = small_df.withColumn("salt", explode(array([lit(i) for i in range(num_salts)])))

df_joined = df_large_salted.join(df_small_exploded, ["skewed_key", "salt"])
```
**Explanation**: Skewed keys cause straggler tasks. Salting distributes hot keys over multiple partitions; after the join, you can re-aggregate to remove salt.

---

### 19) Coalesce multiple columns to the first non-null value
```python
from pyspark.sql.functions import coalesce, col

df_clean = df.withColumn("primary_contact", coalesce(col("mobile"), col("home_phone"), col("email")))
```
**Explanation**: `coalesce` returns the first non-null argument—useful for survivorship rules and canonicalization.

---

### 20) When and how to use a Python UDF
```python
# Prefer native functions for performance (Catalyst optimization & Tungsten codegen).
# Use UDFs only when logic cannot be expressed with built-in functions.
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType

def complex_logic(val):
    return val.upper() if val else "UNKNOWN"

my_udf = udf(complex_logic, StringType())
df_udf = df.withColumn("processed", my_udf(col("raw_data")))
```
**Explanation**: Python UDFs execute outside Spark’s JVM and incur (de)serialization overhead. Consider **SQL expressions**, **built-ins**, or **Pandas UDFs** (vectorized) before standard UDFs. In Spark 3+, **Scala/SQL** or **built-ins** often outperform.

---

### 21) Union two DataFrames with different schemas
```python
# Requires Spark 3.1+
df_union = df1.unionByName(df2, allowMissingColumns=True)
```
**Explanation**: `unionByName` aligns columns by name (not order) and fills missing columns with nulls—key for schema evolution.

---

## Part 4: Delta Lake & Architecture Concepts
Modern data platforms rely on Delta for ACID, schema enforcement, and time travel.

### 22) Write a DataFrame to Delta, partitioned by Year & Month
```python
(
    df.write
      .format("delta")
      .partitionBy("year", "month")
      .mode("append")
      .save("s3://bucket/silver/table")
)
```
**Explanation**: Partitioning improves pruning for time-series queries and reduces I/O. Choose partitions with bounded cardinality to avoid small-file explosion.

---

### 23) Generate a surrogate key for a Gold dimension table
```python
from pyspark.sql.functions import monotonically_increasing_id

# Note: monotonically_increasing_id is unique but not strictly sequential
# For sequential within a partition, use row_number over a deterministic ordering

df_sk = df.withColumn("dim_key", monotonically_increasing_id())
```
**Explanation**: Surrogate keys decouple dimension identity from source natural keys. For strict sequences, handle in a centralized key generator or with windowing + offset logic.

---

### 24) Upsert (MERGE) into a Delta table
```python
from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, "s3://bucket/silver/table")
(
    delta_table.alias("target").merge(
        df_updates.alias("source"),
        "target.id = source.id"
    )
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .execute()
)
```
**Explanation**: Delta MERGE provides ACID-compliant upserts with automatic file and transaction management—essential for CDC, slowly changing dimensions, and idempotent loads.

---

### 25) Time Travel: query an older version of a Delta table
```python
df_v5 = (
    spark.read
         .format("delta")
         .option("versionAsOf", 5)
         .load("s3://bucket/silver/table")
)
```
**Explanation**: Time Travel enables point-in-time queries for reproducibility, auditing, and rollback testing by version or timestamp.

---

### 26) Optimize a Delta table and Z-Order
```python
from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, "s3://bucket/silver/table")
# Compacts small files and reorders data to cluster by frequently filtered columns
(
    delta_table.optimize().executeZOrderBy("customer_id")
)
```
**Explanation**: Compaction mitigates the small-files problem, improving read performance. Z-Ordering co-locates related data pages for faster predicate pruning on high-cardinality columns.

---

## Part 5: Performance Tuning & Streaming
Understanding the engine pays dividends in speed and cost.

### 27) Repartition vs. Coalesce
```python
# Repartition triggers a full shuffle; use to INCREASE partitions or fix skew

df_repart = df.repartition(200)

# Coalesce reduces partitions WITHOUT a shuffle; use to DECREASE partitions (e.g., before writes)

df_coal = df.coalesce(10)
```
**Explanation**: Choose partition counts to balance parallelism and overhead. Repartition for even data distribution; coalesce to reduce file count without the cost of a shuffle.

---

### 28) Persist a DataFrame to memory and disk
```python
from pyspark import StorageLevel

# cache() == persist(StorageLevel.MEMORY_AND_DISK) by default
# Use *_SER levels to store serialized data and reduce memory footprint

df.persist(StorageLevel.MEMORY_AND_DISK_SER)
```
**Explanation**: Persist reused DataFrames to avoid recomputation (lineage traversal). Pick storage levels based on memory pressure and reuse frequency.

---

### 29) Read streaming data from a landing-zone folder
```python
schema = "id INT, name STRING, value DOUBLE"

streaming_df = (
    spark.readStream
         .format("cloudFiles")        # Auto Loader (Databricks)
         .option("cloudFiles.format", "json")
         .schema(schema)
         .load("s3://bucket/landing_zone/")
)
```
**Explanation**: With Auto Loader (Databricks), `cloudFiles` incrementally and efficiently ingests new files with scalable listing. On open-source Spark, use `format("json")` + `readStream` and a known schema.

---

### 30) Write a streaming DataFrame to Delta with checkpointing
```python
query = (
    streaming_df.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", "s3://bucket/checkpoints/my_table")
        .start("s3://bucket/bronze/my_table")
)
```
**Explanation**: Checkpointing persists offsets and progress (WAL) for exactly-once/at-least-once semantics (depending on sink). Writing to Delta provides transactional guarantees for downstream readers.

---

## Bonus: Practical Notes & Gotchas
- **Schema management**: Prefer explicit schemas in production for speed and correctness; set `mergeSchema` only when necessary to handle evolution.
- **Partitioning strategy**: Keep partitions under a few thousand files; monitor small-file issues and compact periodically.
- **Skew & hotspots**: Detect via task timelines and stage skew metrics; treat with salting, AQE (adaptive skew join), or repartitioning.
- **UDF alternatives**: Try SQL expressions, built-ins, `expr`, **Pandas UDFs**, or **Scala UDFs** for better performance.
- **AQE**: Enable Adaptive Query Execution to handle skew joins, coalesce partitions post-shuffle, and optimize plans dynamically.

---

**End of Notebook Markdown**
