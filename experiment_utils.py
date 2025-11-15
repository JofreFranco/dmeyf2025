import logging
import gc
import pandas as pd
import time
import lightgbm as lgb
import psutil
import os
import csv
import joblib
import numpy as np
from dmeyf2025.metrics.revenue import gan_eval

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

def train_model(train_set, params):
    """
    Entrena un modelo Z(uper)LightGBM (lgbm)
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

def print_memory_state():
    try:
        mem = psutil.virtual_memory()
        logger.info(f"Sistema RAM: {mem.percent:.1f}% usado, {mem.available / (1024**3):.2f} GB disponibles, total {mem.total / (1024**3):.2f} GB")
    except Exception as e:
        logger.warning(f"No se pudo leer estado de la RAM: {e}")
def train_models_and_save_results(train_set,X_eval, w_eval, params, seeds, results_file, save_model, n_seeds, experiment_name, fieldnames):
    revs = []
    for seed in seeds[:n_seeds]:
        params["seed"] = seed
        start_time = time.time()
        logger.info(f"Entrenando modelo con seed: {seed}")
        model = train_model(train_set, params)
        y_pred = model.predict(X_eval)
        rev, _ = gan_eval(y_pred, w_eval, window=2001)
        revs.append(rev)
        if rev > 600000000:
            raise Exception(f"Ganancia excesiva: {rev}")
        if rev < 350000000:
            raise Exception(f"Ganancia insuficiente: {rev}")

        write_header = not os.path.exists(results_file)
        
        if save_model:
            joblib.dump(model, f"/home/martin232009/buckets/b1/models/{experiment_name}_{seed}.pkl")
            save_model = False
        end_time = time.time()
        result_row = {
            "experiment_name": experiment_name,
            "seed": seed,
            "training_time": end_time - start_time,
            "moving_average_rev": rev
        }
        with open(results_file, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(result_row)
        gc.collect()
        logger.info(f"Modelo entrenado en tiempo: {end_time - start_time}")
        logger.info(f"Ganancia: {rev}")
    logger.info(f"Ganancias: {revs}")
    logger.info(f"Ganancia promedio: {np.mean(revs)}")
    logger.info(f"Ganancia máxima: {np.max(revs)}")
    logger.info(f"Ganancia mínima: {np.min(revs)}")
    logger.info(f"Ganancia std: {np.std(revs)}")
    logger.info(f"Ganancia mediana: {np.median(revs)}")
    return revs