from pyspark.sql import SparkSession
from pyspark.sql import functions as func


def build_spark():
    return SparkSession.builder \
        .appName("FeatureEngineering") \
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

def aggregate_bureau(bureau):
    return bureau.groupBy("SK_ID_CURR").agg(
        func.count("SK_ID_BUREAU").alias("bureau_loan_count"),
        func.sum("AMT_CREDIT_SUM").alias("bureau_total_credit"),
        func.avg("AMT_CREDIT_SUM").alias("bureau_avg_credit"),
        func.sum("AMT_CREDIT_SUM_DEBT").alias("bureau_total_debt"),
        func.avg("CREDIT_DAY_OVERDUE").alias("bureau_avg_overdue"),
        func.max("CREDIT_DAY_OVERDUE").alias("bureau_max_overdue"),
        func.countDistinct("CREDIT_TYPE").alias("bureau_credit_types"),
        func.sum(func.when(func.col("CREDIT_ACTIVE") == "Active", 1).otherwise(0))
            .alias("bureau_active_loans"),
    )

def add_ratio_features(df):
    return df \
        .withColumn("credit_income_ratio",
                    func.try_divide(func.col("AMT_CREDIT"),
                                    func.col("AMT_INCOME_TOTAL"))) \
        .withColumn("annuity_income_ratio",
                    func.try_divide(func.col("AMT_ANNUITY"),
                                    func.col("AMT_INCOME_TOTAL"))) \
        .withColumn("credit_term",
                    func.try_divide(func.col("AMT_CREDIT"),
                                    func.col("AMT_ANNUITY"))) \
        .withColumn("days_employed_ratio",
                    func.try_divide(func.col("DAYS_EMPLOYED"),
                                    func.col("DAYS_BIRTH"))) \
        .withColumn("debt_credit_ratio",
                    func.try_divide(func.col("bureau_total_debt"),
                                    func.col("bureau_total_credit")))

def main():
    spark = build_spark()

    application = spark \
        .read.format("hudi") \
        .load(
            f"hdfs://hdfs-namenode:9000/project/cleaned/application_train"
        )
    bureau = spark \
        .read.format("hudi") \
        .load(
            f"hdfs://hdfs-namenode:9000/project/cleaned/bureau"
        )

    hoodie_cols = [c for c in application.columns if c.startswith("_hoodie_")]
    application = application.drop(*hoodie_cols)

    hoodie_cols_b = [c for c in bureau.columns if c.startswith("_hoodie_")]
    bureau = bureau.drop(*hoodie_cols_b)

    bureau_agg = aggregate_bureau(bureau)

    df = application.join(bureau_agg, on="SK_ID_CURR", how="left")

    df = add_ratio_features(df)

    hudi_options = {
        "hoodie.table.name": "home_credit_features",
        "hoodie.datasource.write.recordkey.field": "SK_ID_CURR",
        "hoodie.datasource.write.table.name": "home_credit_features",
        "hoodie.datasource.write.operation": "insert",
        "hoodie.datasource.write.precombine.field": "SK_ID_CURR",
        "hoodie.datasource.hive_sync.enable": "false",
    }

    df.write.format("hudi") \
        .options(**hudi_options) \
        .mode("overwrite") \
        .save("hdfs://hdfs-namenode:9000/project/features")

    spark.stop()


if __name__ == "__main__":
    main()
