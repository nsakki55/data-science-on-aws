import argparse
import json
import os
import time
from datetime import datetime
from logging import getLogger
from typing import Any, Dict

import boto3
import numpy as np
import pandas as pd
import sagemaker
import yaml
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split

logger = getLogger(__name__)

feature_cols = [
    "id",
    "event_time",
    "click",
    "hour",
    "C1",
    "banner_pos",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "device_type",
    "device_conn_type",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
]
target = "click"

SETTING_FILE_PATH = "../settings.yaml"
with open(SETTING_FILE_PATH) as file:
    aws_info = yaml.safe_load(file)

sess = sagemaker.Session()
role = aws_info["aws"]["sagemaker"]["role"]
bucket = aws_info["aws"]["sagemaker"]["s3bucket"]
region = boto3.Session().region_name

sm = boto3.client(service_name="sagemaker")
session = boto3.Session(region_name=region)
featurestore_runtime = boto3.client(service_name="sagemaker-featurestore-runtime")

s3 = boto3.client(service_name="s3")

sagemaker_session = sagemaker.Session(
    boto_session=session, sagemaker_client=sm, sagemaker_featurestore_runtime_client=featurestore_runtime
)


def wait_for_feature_group_creation_complete(feature_group):
    try:
        status = feature_group.describe().get("FeatureGroupStatus")
        logger.info("Feature Group status: {}".format(status))
        while status == "Creating":
            logger.info("Waiting for Feature Group Creation")
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
            logger.info("Feature Group status: {}".format(status))
        if status != "Created":
            logger.info("Feature Group status: {}".format(status))
            raise RuntimeError(f"Failed to create feature group {feature_group.name}")
        logger.info(f"FeatureGroup {feature_group.name} successfully created.")
    except:
        logger.info("No feature group created yet.")


def create_or_load_feature_group(prefix: str, feature_group_name: str) -> FeatureGroup:
    prefix = "ctr-prediction-feature-store"

    feature_definitions = [
        FeatureDefinition(feature_name=feature_name, feature_type=FeatureTypeEnum.STRING) for feature_name in feature_cols
    ]

    feature_group = FeatureGroup(
        name=feature_group_name, feature_definitions=feature_definitions, sagemaker_session=sagemaker_session
    )
    logger.info("Feature Group: {}".format(feature_group))

    try:
        logger.info("Waiting for existing Feature Group to become available")
        wait_for_feature_group_creation_complete(feature_group)
    except Exception as e:
        logger.info("Before CREATE FG wait exeption: {}".format(e))

    try:
        record_identifier_name = "id"
        event_time_feature_name = "event_time"

        logger.info("Creating Feature Group with role {}...".format(role))
        feature_group.create(
            s3_uri=f"s3://{bucket}/{prefix}",
            record_identifier_name=record_identifier_name,
            event_time_feature_name=event_time_feature_name,
            role_arn=role,
            enable_online_store=False,
        )
        logger.info("Creating Feature Group. Completed.")

        logger.info("Waiting for new Feature Group to become available...")
        wait_for_feature_group_creation_complete(feature_group)
        logger.info("Feature Group available.")
        feature_group.describe()

    except Exception as e:
        logger.info("Exception: {}".format(e))

    return feature_group


def preprocess(df: pd.DataFrame):
    df["hour"] = df["hour"].map(lambda x: datetime.strptime(str(x), "%y%m%d%H"))
    df["day_of_week"] = df["hour"].map(lambda x: x.hour)

    feature_hasher = FeatureHasher(n_features=2**24, input_type="string")
    hashed_feature = feature_hasher.fit_transform(np.asanyarray(df.astype(str)))

    return hashed_feature


def list_arg(raw_value):
    """argparse type for a list of strings"""
    return str(raw_value).split(",")


def parse_agrs() -> Dict[str, Any]:
    resconfig = {}
    try:
        with open("/opt/ml/config/resourceconfig.json", "r") as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print("/opt/ml/config/resourceconfig.json not found.  current_host is unknown.")
        pass  # Ignore

    parser = argparse.ArgumentParser(description="sagemaker-processor")

    parser.add_argument(
        "--hosts",
        type=list_arg,
        default=resconfig.get("hosts", ["unknown"]),
        help="Comma-separated list of host names running the job",
    )
    parser.add_argument(
        "--current-host",
        type=str,
        default=resconfig.get("current_host", "unknown"),
        help="Name of this host running the job",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="/opt/ml/processing/input/data",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="/opt/ml/processing/output",
    )
    parser.add_argument(
        "--train-split-percentage",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--validation-split-percentage",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--test-split-percentage",
        type=float,
        default=0.05,
    )
    parser.add_argument("--balance-dataset", type=eval, default=True)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--feature-store-offline-prefix",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--feature-group-name",
        type=str,
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_agrs()
    input_data_path = os.path.join("/opt/ml/processing/input", "train.csv")
    df = pd.read_csv(input_data_path)
    logger.info(f"DataFrame: {df.shape}")

    current_time_sec = int(round(time.time()))
    df["event_time"] = pd.Series([current_time_sec] * len(df), dtype="float64")

    y_train = df[target].values
    y_train = np.asarray(y_train).ravel()
    X_train = df[feature_cols]
    X_train_hashed = preprocess(X_train)

    df_train = X_train.to_frame().copy()
    df_train["click"] = y_train
    df_train.to_csv("/opt/ml/processing/output/train/train.csv")
