# Technical challenge - Home Credit Default Risk

A bare-metal DataOps platform. It ingests the Home Credit Default Risk dataset, transforms it with Apache Spark, stores it as Apache Hudi tables on Apache HDFS, and orchestrates the whole flow with Apache Airflow.
The bonus task trains an XGBoost model to predict loan default.

Everything runs on virtual machines on a Proxmox hypervisor.

## Challenge Requirements → Deliverables

| Requirement                                                   | Delivered                                                        |
| ------------------------------------------------------------- | ---------------------------------------------------------------- |
| Environment: Hadoop (HDFS) + Spark + Airflow                  | Bare-metal VM cluster                                            |
| Process Home Credit dataset with Spark                        | `sanitize_data.py` + `feature_engineering.py`                    |
| Convert at least one table to an open table format            | Hudi tables for application_train, bureau, and the feature table |
| Airflow DAG (ingest → transform → write)                      | `kaggle_airflow_spark_hdfs.py`                                   |
| Architecture diagram                                          | `architecture_diagram.drawio.png` (Draw.io)                      |
| **Bonus:** join tables, feature engineering, XGBoost, ROC-AUC | `home_credit_xgboost.ipynb`                                      |

## Architecture

```
                    ┌────────────────────────────────────────────┐
                    │  Apache Airflow (LocalExecutor + Postgres) │
                    │  orchestrates the pipeline (control flow)  │
                    └────────────────────────────────────────────┘
                                      ┊ triggers
                                      ▼
  ┌────────────┐   ┌────────────────────────────────────┐   ┌───────────────────────┐
  │ Kaggle API │──▶│        Apache HDFS (3.4.1)         │◀─▶│  Apache Spark (4.0.2) │
  │            │   │  NameNode + DataNode               │   │  Master + 2 Workers   │
  └────────────┘   │                                    │   │                       │
                   │  /project/rawdata                  │   │ sanitize_data.py      │
                   │  /project/cleaned                  │   │ feature_engineering.py│
                   │  /project/features                 │   └───────────────────────┘
                   └────────────────────────────────────┘
                              │ reads /project/features
                              ▼
                   ┌────────────────────────────────────┐
                   │  ★ XGBoost notebook (bonus)        │
                   │  predicts TARGET, ROC-AUC ≈ 0.76   │
                   └────────────────────────────────────┘
```

Data flows raw → cleaned → features → model.
Airflow controls the flow but delegates all heavy processing to the Spark cluster.

## Technology Stack

| Component            | Version | Role                             |
| -------------------- | ------- | -------------------------------- |
| Proxmox VE           | —       | Hypervisor                       |
| Ubuntu Server        | 22.04   | Guest OS                         |
| Apache Hadoop (HDFS) | 3.4.1   | Distributed storage              |
| Apache Spark         | 4.0.2   | Distributed processing           |
| Apache Airflow       | 2.9.3   | Orchestration                    |
| Apache Hudi          | 1.1.1   | Open table format                |
| PostgreSQL           | 16      | Airflow metadata DB              |
| XGBoost              | 2.x     | Default-prediction model (bonus) |
| Java                 | 17      | Runtime for Hadoop/Spark         |

Hadoop 3.4.1 was chosen to match the Hadoop client libraries that Spark 4.0.2 already bundles, which avoids JAR/version conflicts when Spark reads and writes HDFS.

## VM Layout

| VM             | Role                      | vCPU | RAM  | Disk   | IP             |
| -------------- | ------------------------- | ---- | ---- | ------ | -------------- |
| hdfs-namenode  | HDFS NameNode + ingestion | 2    | 4 GB | 30 GB  | 192.168.100.20 |
| hdfs-datanode  | HDFS DataNode             | 2    | 6 GB | 100 GB | 192.168.100.21 |
| spark-master   | Spark Master              | 2    | 4 GB | 30 GB  | 192.168.100.12 |
| spark-worker-1 | Spark Worker              | 4    | 8 GB | 30 GB  | 192.168.100.13 |
| spark-worker-2 | Spark Worker              | 4    | 8 GB | 30 GB  | 192.168.100.16 |
| airflow        | Airflow + PostgreSQL      | 2    | 4 GB | 30 GB  | 192.168.100.11 |

All VMs share an isolated cluster bridge (`vmbr1`, 192.168.100.0/24) for internal traffic and a second interface for internet access. All cluster hostnames are in every VM's `/etc/hosts`, and passwordless SSH is configured between the nodes that need it (NameNode→DataNode, Spark master→workers, Airflow→NameNode/Spark).

## Setup

The full per-component installation (Hadoop, Spark, Airflow) is summarized below; each was installed under a dedicated service user (`hadoop`, `spark`, `airflow`) owning its `/opt/<tool>` install.

### HDFS (Hadoop 3.4.1)

On both HDFS VMs: install Java 17, create the `hadoop` user, extract Hadoop to `/opt/hadoop`, set `JAVA_HOME` in `hadoop-env.sh`, and configure `core-site.xml` and `hdfs-site.xml`.
The `workers` file on the NameNode lists `hdfs-datanode`.
Format the NameNode once (`hdfs namenode -format`), then `start-dfs.sh`.

### Spark (4.0.2)

Standalone cluster: master on `spark-master`, workers listed in `conf/workers`.
Copy HDFS's `core-site.xml` and `hdfs-site.xml` into `/opt/spark/conf/` so Spark can reach HDFS.
Place the Hudi bundle JAR in `/opt/spark/jars/`.
No extra Hadoop JARs are needed because Spark 4.0.2 bundles Hadoop 3.4.1 client libraries.

### Airflow (2.9.3)

Install into a Python venv **always using the version constraints file**, with a PostgreSQL backend and `LocalExecutor`:

```bash
pip install "apache-airflow==2.9.3" "apache-airflow-providers-apache-spark" \
  "apache-airflow-providers-postgres" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.12.txt"
```

## Running the Pipeline

### 1. Start the cluster (in order)

```bash
# HDFS (on hdfs-namenode, as hadoop)
start-dfs.sh
hdfs dfsadmin -report            # verify one live datanode

# Spark (on spark-master)
$SPARK_HOME/sbin/start-all.sh

# Airflow (on airflow VM, in venv)
airflow scheduler &
airflow dag-processor &
airflow triggerer &
airflow webserver --port 8090 &

or run the start_airflow.sh script
```

### 2. Trigger the DAG

From the UI (tunnel `-L 8090:192.168.100.11:8090`, open `http://localhost:8090`) or the CLI:

```bash
airflow dags trigger home_credit_hdfs_pipeline
```

Task order: `download_kaggle_data → init_hdfs → upload_raw_to_hdfs → sanitize_data → feature_engineering → cleanup_local`.

### 3. Verify output

```bash
hdfs dfs -ls /project/rawdata/    # all raw CSVs
hdfs dfs -ls /project/cleaned/    # application_train, bureau (Hudi)
hdfs dfs -ls /project/features/   # feature table (Hudi)
```

### 4. Run the bonus ML notebook (standalone)

```bash
# on spark-master
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
# tunnel -L 8888:192.168.100.12:8888, open the notebook, run top to bottom
```

## Bonus task: Model Results

The XGBoost model, trained on `application_train` joined with aggregated `bureau` plus engineered ratio features:

- **ROC-AUC ≈ 0.76** (primary metric — threshold-independent and robust to the ~92/8 class imbalance)
- **Recall on defaulters ≈ 0.66** — catches about two-thirds of true defaulters
- Imbalance handled with `scale_pos_weight` (≈ 11.4); evaluated with ROC-AUC plus a precision/recall and confusion-matrix view

Accuracy (≈ 0.72) is intentionally lower than the 92% "predict everyone repays" baseline which catches zero defaulters, which is why ROC-AUC, not accuracy, is the reported metric.
The result is a solid two-table baseline.
