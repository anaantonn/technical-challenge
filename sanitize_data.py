from pyspark.sql import SparkSession
from pyspark.sql import functions as func


TABLES = {
    "application_train": "SK_ID_CURR",
    "bureau": "SK_ID_BUREAU",
}


def build_spark():
    return SparkSession.builder \
        .appName("SanitizeData") \
        .config(
            "spark.serializer",
            "org.apache.spark.serializer.KryoSerializer"
        ) \
        .config(
            "spark.sql.extensions",
            "org.apache.spark.sql.hudi.HoodieSparkSessionExtension"
        ) \
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.hudi.catalog.HoodieCatalog"
        ) \
        .config(
            "spark.hadoop.fs.defaultFS",
            "hdfs://hdfs-namenode:9000"
        ) \
        .getOrCreate()


def sanitize_table(df):
    # Separate columns by type
    string_cols = [c for c, t in df.dtypes if t == "string"]

    # Trim whitespace on all string columns
    for c in string_cols:
        df = df.withColumn(c, func.trim(func.col(c)))

    # Fill string nulls with "Unknown".
    if string_cols:
        df = df.fillna("Unknown", subset=string_cols)

    return df


def main():
    spark = build_spark()

    for table_name, record_key in TABLES.items():
        df = spark.read.csv(
            f"hdfs://hdfs-namenode:9000/project/rawdata/{table_name}.csv",
            header=True,
            inferSchema=True,
        )

        cleaned = sanitize_table(df)

        hudi_options = {
            "hoodie.table.name": table_name,
            "hoodie.datasource.write.recordkey.field": record_key,
            "hoodie.datasource.write.table.name": table_name,
            "hoodie.datasource.write.operation": "insert",
            "hoodie.datasource.write.precombine.field": record_key,
            "hoodie.datasource.hive_sync.enable": "false",
        }

        cleaned.write.format("hudi") \
            .options(**hudi_options) \
            .mode("overwrite") \
            .save(f"hdfs://hdfs-namenode:9000/project/cleaned/{table_name}")

    spark.stop()


if __name__ == "__main__":
    main()
