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

# Algunos settings
VERBOSE = False
experiment_name = "zlgbm-baseline"
training_months = [201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908,
       201909, 201910, 201911, 201912, 202001, 202002, 202003, 202004,
       202005, 202006, 202007, 202008, 202009, 202010, 202011, 202012,
       202101, 202102, 202103, 202104]
save_model = True
eval_month = 202106
test_month = 202108
seeds = [537919, 923347, 173629, 419351, 287887, 1244, 24341, 1241, 4512, 6554, 62325, 6525235, 14, 4521, 474574, 74543, 32462, 12455, 5124, 55678]
debug_mode = True
sampling_rate = 0.1
results_file = "/home/martin232009/buckets/b1/results.csv"
fieldnames = ["experiment_name", "seed", "training_time", "moving_average_rev"]
logging.info("comenzando")
features_to_drop = ["cprestamos_prendarios", "mprestamos_prendarios", "cprestamos_personales", "mprestamos_personales"]
canaritos = 10
gradient_bound = 0.1
n_seeds = 5
params = {
    "canaritos": canaritos,
    "gradient_bound": gradient_bound,
    "feature_fraction": 0.50,
    "min_data_in_leaf": 20,
    "is_unbalance": True,
}
if debug_mode:
    n_seeds = 1
    sampling_rate = 0.01
    params["min_data_in_leaf"] = 2000
    params["gradient_bound"] = 0.4
experiment_name = f"{experiment_name}_c{canaritos}_gb{experiment_name}_s{sampling_rate}_u{(params['is_unbalance'])}"
def memory_gb(df: pd.DataFrame) -> float:
    return df.memory_usage().sum() / (1024 ** 3)

def apply_transformer(transformer, X, name: str, logger):
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
    initial_cols =X_transformed.columns
    X_transformed = apply_transformer(
        CleanZerosTransformer(),
        X_transformed,
        "CleanZerosTransformer",
        logger
    )
    new_cols = list(set(X_transformed.columns) - set(initial_cols))
    X_transformed = apply_transformer(
        DeltaLagTransformer(
            n_lags=2,
            exclude_cols=["foto_mes","numero_de_cliente","target","label","weight","clase_ternaria"] + new_cols
        ),
        X_transformed,
        "DeltaLagTransformer",
        logger
    )

    X_transformed = apply_transformer(
        PeriodStatsTransformer(
            periods=[6],
            exclude_cols=["foto_mes","numero_de_cliente","target","label","weight","clase_ternaria"] + new_cols),
        X_transformed,
        "PeriodStats",
        logger
    )
        
    X_transformed = apply_transformer(
        TendencyTransformer(),
        X_transformed,
        "PeriodStats",
        logger
    )
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


# Leer datos
logger.info("Leyendo dataset")
df = pd.read_csv('~/datasets/competencia_02_target.csv')
# Eliminar features que no se van a usar
keep_cols = [col for col in df.columns if col not in features_to_drop]
df = df[keep_cols]
df = df[~df["foto_mes"].isna()]
# Agregar target y calcular weight
weight = {"BAJA+1": 1, "BAJA+2": 1.00002, "CONTINUA": 1}
df["target"] = ((df["clase_ternaria"] == "BAJA+2") | (df["clase_ternaria"] == "BAJA+1")).astype(int)

# Preparar datos
start_time = time.time()
logger.info("Preparando datos")
X_train, y_train, w_train, X_eval, y_eval, w_eval, X_test, y_test = prepare_data(df, training_months, eval_month, test_month, get_features, weight, sampling_rate)
del df
gc.collect()
try:
    mem = psutil.virtual_memory()
    logger.info(f"Sistema RAM: {mem.percent:.1f}% usado, {mem.available / (1024**3):.2f} GB disponibles, total {mem.total / (1024**3):.2f} GB")
except Exception as e:
    logger.warning(f"No se pudo leer estado de la RAM: {e}")
logger.info("Agregando canaritos")
X_train = AddCanaritos(n_canaritos=canaritos).fit_transform(X_train)
X_eval = AddCanaritos(n_canaritos=canaritos).fit_transform(X_eval) 
logger.info("Datos pre procesados en tiempo: %s", time.time() - start_time)
# loggear uso de memoria del dataset
try:
    mem_gb = X_train.memory_usage().sum() / (1024 ** 3)
    logger.info(f"Uso de memoria del dataset: {mem_gb:.2f} GB")
except:
    logger.info("No se pudo calcular el uso de memoria del dataset")
revs = []
train_set = lgb.Dataset(X_train, label=y_train)
for seed in seeds[:n_seeds]:
    params["seed"] = seed
    start_time = time.time()
    logger.info(f"Entrenando modelo con seed: {seed}")
    model = train_model(train_set, params)
    try:
        y_pred = model.predict(X_eval)
        rev, _ = gan_eval(y_pred, w_eval, window=2001)
        revs.append(rev)
    except Exception as e:
        logger.info(y_pred.shape)
        logger.info(y_pred)
        logger.info(f"time: {time.time()- start_time}")
        logger.error(f"Error al predecir: {e}")
        raise e

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
# Eliminar features con canaritos
# TODO: Implementar
