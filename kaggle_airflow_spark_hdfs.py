from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 6, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "home_credit_hdfs_pipeline",
    default_args=default_args,
    description="End-to-end: Kaggle → Spark → HDFS pipeline",
    schedule=None,
    catchup=False,
    tags=["home-credit", "hdfs", "spark", "hudi"],
)

# Task 1: download the dataset from Kaggle onto the NameNode VM
download_kaggle_data = BashOperator(
    task_id="download_kaggle_data",
    bash_command=f"""
    set -e
    ssh hadoop@192.168.100.20 '
        export KAGGLE_CONFIG_DIR=/home/hadoop/.config/kaggle
        rm -rf /tmp/home_credit_raw
        mkdir -p /tmp/home_credit_raw
        cd /tmp/home_credit_raw
        /home/hadoop/kaggle-venv/bin/kaggle competitions download -c home-credit-default-risk --force
        unzip -o "*.zip"
        rm -f *.zip
        echo "Download complete"
    '
    """,
    dag=dag,
)

# Task 2: create the HDFS directory layout
init_hdfs = BashOperator(
    task_id="init_hdfs",
    bash_command=f"""
    ssh hadoop@192.168.100.20 '
        hdfs dfs -mkdir -p /project/rawdata
        hdfs dfs -mkdir -p /project/cleaned
        hdfs dfs -mkdir -p /project/features
        hdfs dfs -chmod -R 777 /project
    '
    """,
    dag=dag,
)

# Task 3: put all raw CSVs into HDFS
upload_raw_to_hdfs = BashOperator(
    task_id="upload_raw_to_hdfs",
    bash_command=f"""
    ssh hadoop@192.168.100.20 '
        for f in /tmp/home_credit_raw/*.csv; do
            fname=$(basename "$f")
            hdfs dfs -put -f "$f" /project/rawdata/$fname
            echo "Uploaded $fname"
        done
    '
    """,
    dag=dag,
)

# Task 4: Spark sanitization (application_train + bureau -> /project/cleaned)
sanitize_data = BashOperator(
    task_id="sanitize_data",
    bash_command=f"""
    ssh spark@192.168.100.12 '
        /opt/spark/bin/spark-submit \
            --master spark://192.168.100.12:7077 \
            --jars /opt/spark/jars/hudi-spark4.0-bundle_2.13-1.1.1.jar \
            /opt/spark/jobs/sanitize_data.py
    '
    """,
    dag=dag,
)

# Task 5: Spark feature engineering (join + features -> /project/features)
feature_engineering = BashOperator(
    task_id="feature_engineering",
    bash_command=f"""
    ssh spark@192.168.100.12 '
        /opt/spark/bin/spark-submit \
            --master spark://192.168.100.12:7077 \
            --jars /opt/spark/jars/hudi-spark4.0-bundle_2.13-1.1.1.jar \
            /opt/spark/jobs/feature_engineering.py
    '
    """,
    dag=dag,
)

# Task 6: clean up local files
cleanup_local = BashOperator(
    task_id="cleanup_local",
    bash_command=f"""
    ssh hadoop@192.168.100.20 'rm -rf /tmp/home_credit_raw/*'
    """,
    dag=dag,
)


download_kaggle_data >> init_hdfs >> upload_raw_to_hdfs >> sanitize_data >> feature_engineering >> cleanup_local
