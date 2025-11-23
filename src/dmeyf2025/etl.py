import logging
from os.path import exists
import pandas as pd
import os
import gc
import numpy as np
from sklearn.pipeline import Pipeline
from dmeyf2025.processors.sampler import SamplerProcessor
logger = logging.getLogger(__name__)


def prepare_data(df, training_months, eval_month, test_month, get_features, weight, **sampler_conf):
    df["label"] = ((df["clase_ternaria"] == "BAJA+2") | (df["clase_ternaria"] == "BAJA+1")).astype(int)
    df["weight"] = np.array([weight[item] for item in df["clase_ternaria"]])
    df = df.drop(columns=["clase_ternaria"])
    df_transformed = get_features(df, training_months)
    del df
    gc.collect()
    if training_months is not None:
        df_train = df_transformed[df_transformed["foto_mes"].isin(training_months)]
    else:
        df_train = df_transformed[~df_transformed["foto_mes"] < eval_month]
    df_eval = df_transformed[df_transformed["foto_mes"] == eval_month]
    df_test = df_transformed[df_transformed["foto_mes"] == test_month]
    del df_transformed
    gc.collect()
    y_eval, w_eval, X_eval = df_eval["label"], df_eval["weight"], df_eval.drop(columns=["label", "weight"])
    del df_eval
    gc.collect()
    y_test, w_test, X_test = df_test["label"], df_test["weight"], df_test.drop(columns=["label", "weight"])
    del df_test
    gc.collect()
    sampler = SamplerProcessor(**sampler_conf)
    df_sampled = sampler.transform(df_train)
    X_train, y_train = df_sampled.drop(columns=["label"]), df_sampled["label"]
    del df_train, df_sampled
    gc.collect()
    w_train = X_train["weight"]
    X_train = X_train.drop(columns=["weight"])
    if "label" in X_train.columns:
        X_train = X_train.drop(columns=["label"])
    if "weight" in X_train.columns:
        X_train = X_train.drop(columns=["weight"])
    if "clase_ternaria" in X_train.columns:
        X_train = X_train.drop(columns=["clase_ternaria"])
    if "numero_de_cliente" in X_train.columns:
        X_train = X_train.drop(columns=["numero_de_cliente"])

    if "label" in X_test.columns:
        X_test = X_test.drop(columns=["label"])
    if "weight" in X_test.columns:
        X_test = X_test.drop(columns=["weight"])
    if "clase_ternaria" in X_test.columns:
        X_test = X_test.drop(columns=["clase_ternaria"])
    if "numero_de_cliente" in X_test.columns:
        X_test = X_test.drop(columns=["numero_de_cliente"])
    if "label" in X_eval.columns:
        X_eval = X_eval.drop(columns=["label"])
    if "weight" in X_eval.columns:
        X_eval = X_eval.drop(columns=["weight"])
    if "clase_ternaria" in X_eval.columns:
        X_eval = X_eval.drop(columns=["clase_ternaria"])
    if "numero_de_cliente" in X_eval.columns:
        X_eval = X_eval.drop(columns=["numero_de_cliente"])
        
    return X_train, y_train, w_train, X_eval, y_eval, w_eval, X_test, y_test
if __name__ == "__main__":
    pass