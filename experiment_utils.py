import logging
import os
import csv
import joblib
import psutil
import gc
import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from dmeyf2025.processors.feature_processors import CleanZerosTransformer, DeltaLagTransformer, PercentileTransformer, PeriodStatsTransformer, TendencyTransformer, IntraMonthTransformer, RandomForestFeaturesTransformer, DatesTransformer, HistoricalFeaturesTransformer, AddCanaritos

from dmeyf2025.metrics.revenue import gan_eval
from dmeyf2025.etl.etl import prepare_data
pd.set_option('display.max_columns', None)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Para mostrar en consola
    ]
)
def memory_gb(df: pd.DataFrame) -> float:
    return df.memory_usage().sum() / (1024 ** 3)

def apply_transformer(transformer, X, name: str, logger, VERBOSE=False):
    logger.info(f"[{name}] Iniciando…")

    start_mem = memory_gb(X)
    start_time = time.time()

    Xt = transformer.fit_transform(X)

    end_time = time.time()
    end_mem = memory_gb(Xt)

    n_rows, n_cols = Xt.shape

    logger.info(
        f"[{name}] Tiempo: {end_time - start_time:.2f}s | "
        f"Memoria antes: {start_mem:.3f} GB | "
        f"Memoria después: {end_mem:.3f} GB | "
        f"Diferencia: {end_mem - start_mem:+.3f} GB | "
        f"Shape: {n_rows:,} filas × {n_cols:,} columnas"
    )
    if VERBOSE:
        display(Xt.head())
        display(Xt.describe())
        logger.info(f"Nulos: {Xt.isna().astype(int).sum()}")
    gc.collect()
    return Xt


def get_features(X, training_months):

    X_transformed = X

    X_transformed = apply_transformer(
        CleanZerosTransformer(),
        X_transformed,
        "CleanZerosTransformer",
        logger
    )

    X_transformed = apply_transformer(
        DeltaLagTransformer(
            n_lags=2,
            exclude_cols=["foto_mes","numero_de_cliente","target","label","weight","clase_ternaria"]
        ),
        X_transformed,
        "DeltaLagTransformer",
        logger
    )
    logger.info(f"Cantidad de features después de delta lag transformer: {len(X_transformed.columns)}")

    X_transformed = apply_transformer(
        PercentileTransformer(
            replace_original=True
        ),
        X_transformed,
        "PercentileTransformer",
        logger
    )

    return X_transformed



def train_model(train_set, params):
    """
    Entrena un modelo ZuperLightGBM (lgbm)
    Args:
        X_train (pd.DataFrame): Features de entrenamiento
        y_train (pd.Series): Variable objetivo de entrenamiento
        w_train (pd.Series): Weights
        params (dict): diccionario que debe tener:
            - 'semilla_primigenia'
            - 'min_data_in_leaf'
            - 'learning_rate'
            - 'canaritos': maneja el overfitting mediante canaritos, cuando detecta un árbol cuyo primer split es un canarito lo mata.
            - 'gradient_bound': bound para el gradiente es algo asi como un learning rate que va cambiando a medida que se va entrenando???.
    """
    lgb_params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "None",        # Para usar métrica custom
        "first_metric_only": False,
        "boost_from_average": True,
        "feature_pre_filter": False,
        "force_row_wise": True,
        "verbosity": -100,
        "seed": params["seed"],

        "max_bin": 31,
        "min_data_in_leaf": params["min_data_in_leaf"],

        "num_iterations": 9999,
        "num_leaves": 9999,
        "learning_rate": 1,

        "feature_fraction": params["feature_fraction"],

        # Hiperparámetros del Zuperlightgbm
        "canaritos": params["canaritos"],
        "gradient_bound": params["gradient_bound"],  
    }

    
    gbm = lgb.train(
        lgb_params,
        train_set
    )
    return gbm