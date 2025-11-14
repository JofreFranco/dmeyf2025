import logging
import os
import csv
import joblib
import gc
import pandas as pd
import numpy as np
import time
from dmeyf2025.processors.feature_processors import CleanZerosTransformer, DeltaLagTransformer, PercentileTransformer, PeriodStatsTransformer, TendencyTransformer, IntraMonthTransformer, RandomForestFeaturesTransformer, DatesTransformer, HistoricalFeaturesTransformer, AddCanaritos
from dmeyf2025.metrics.revenue import GANANCIA_ACIERTO, COSTO_ESTIMULO
from dmeyf2025.processors.sampler import SamplerProcessor
from dmeyf2025.metrics.revenue import gan_eval
from dmeyf2025.etl.etl import prepare_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Para mostrar en consola
    ]
)

# Algunos settings
experiment_name = "zlgbm-baseline"
training_months = None # todos los meses
save_model = True
eval_month = 202106
test_month = 202108
seeds = [537919, 923347, 173629, 419351, 287887, 1244, 24341, 1241, 4512, 6554, 62325, 6525235, 14, 4521, 474574, 74543, 32462, 12455, 5124, 55678]
debug_mode = False
sampling_rate = 0.1
results_file = "/home/martin232009/buckets/b1/results.csv"
fieldnames = ["experiment_name", "seed", "training_time", "moving_average_rev"]
logging.info("comenzando")
features_to_drop = ["cprestamos_prendarios", "mprestamos_prendarios", "cprestamos_personales", "mprestamos_personales"]
canaritos = 10
grouding_bound = 0.1
n_seeds = 5
params = {
    "canaritos": canaritos,
    "grouding_bound": grouding_bound,
    "feature_fraction": 0.50,
    "min_data_in_leaf": 20,
}
def get_features(X, training_months):
    logger.info("Iniciando clean zeros transformer...")
    clean_zeros_transformer = CleanZerosTransformer()
    X_transformed = clean_zeros_transformer.fit_transform(X)
    logger.info("Iniciando delta lag transformer...")
    delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2, exclude_cols= ["foto_mes", "numero_de_cliente", "target", "label", "weight", "clase_ternaria"])
    X_transformed = delta_lag_transformer.fit_transform(X_transformed)
    logger.info(f"Cantidad de features después de delta lag transformer: {len(X_transformed.columns)}")
    # Percentiles discretizados en saltos de None
    logger.info("Iniciando percentiles transformer...")
    percentiles_transformer = PercentileTransformer(n_bins=None, replace_original=True)
    original_columns = len(X_transformed.columns)
    X_transformed = percentiles_transformer.fit_transform(X_transformed)
    if len(X_transformed.columns) != original_columns:
        raise ValueError(f"Cantidad de features después de percentiles transformer: {len(X_transformed.columns)} != {original_columns}")
    
    return X_transformed


def train_model(X_train, y_train, w_train, params):
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

    train_set = lgb.Dataset(X_train, label=y_train, weight=w_train)
    gbm = lgb.train(
        lgb_params,
        train_set,
        verbose_eval=False
    )
    return gbm

# Leer datos
logger.info("Leyendo dataset")
df = pd.read_csv('~/datasets/competencia_02_target.csv')
# Eliminar features que no se van a usar
keep_cols = [col for col in df.columns if col not in features_to_drop]
df = df[keep_cols]

# Agregar target y calcular weight
weight = {"BAJA+1": 1, "BAJA+2": 1.00002, "CONTINUA": 1}
df["target"] = ((df["clase_ternaria"] == "BAJA+2") | (df["clase_ternaria"] == "BAJA+1")).astype(int)

# Preparar datos
start_time = time.time()
logger.info("Preparando datos")
X_train, y_train, w_train, X_eval, y_eval, w_eval, X_test, y_test = prepare_data(df, training_months, eval_month, test_month, get_features, weight, sampling_rate)
logger.info("Agregando canaritos")
X_train = AddCanaritos(n_canaritos=canaritos).fit_transform(X_train)

logger.info("Datos pre procesados en tiempo: %s", time.time() - start_time)
# loggear uso de memoria del dataset
try:
    logger.info(f"Uso de memoria del dataset: {X_train.memory_usage().sum()}")
except:
    logger.info("No se pudo calcular el uso de memoria del dataset")

# Eliminar features con canaritos
# TODO: Implementar
revs = []
for seed in seeds[:n_seeds]:
    params["seed"] = seed
    start_time = time.time()
    logger.info(f"Entrenando modelo con seed: {seed}")
    model = train_model(X_train, y_train, w_train, params)
    y_pred = model.predict_proba(X_eval)[:,1]
    rev, _ = gan_eval(y_pred, w_eval, window=2001)
    revs.append(rev)

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
