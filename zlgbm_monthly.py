import logging
import gc
import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from dmeyf2025.processors.feature_processors import CleanZerosTransformer, DeltaLagTransformer, PercentileTransformer, PeriodStatsTransformer, TendencyTransformer, IntraMonthTransformer, RandomForestFeaturesTransformer, DatesTransformer, HistoricalFeaturesTransformer, AddCanaritos
from experiment_utils import *
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

def get_features(X, training_months):

    X_transformed = X
    initial_cols =X_transformed.columns
    X_transformed = apply_transformer(
        CleanZerosTransformer(),
        X_transformed,
        "CleanZerosTransformer",
        logger
    )
    X_transformed = apply_transformer(
        IntraMonthTransformer(
            exclude_cols=["foto_mes","numero_de_cliente","target","label","weight","clase_ternaria"]),
        X_transformed,
        "IntraMonthTransformer",
        logger
    )
    X_transformed = apply_transformer(
            DatesTransformer(),
            X_transformed,
            "PeriodStats",
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
        PercentileTransformer(
            replace_original=True
        ),
        X_transformed,
        "PercentileTransformer",
        logger
    )

    
    return X_transformed

# Algunos settings
VERBOSE = False
experiment_name = "zlgbm-intramonth"
training_months = [201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908,
       201909, 201910, 201911, 201912, 202001, 202002, 202003, 202004,
       202005, 202006, 202007, 202008, 202009, 202010, 202011, 202012,
       202101, 202102, 202103, 202104]
save_model = True
eval_month = 202106
test_month = 202108
seeds = [537919, 923347, 173629, 419351, 287887, 1244, 24341, 1241, 4512, 6554, 62325, 6525235, 14, 4521, 474574, 74543, 32462, 12455, 5124, 55678]
debug_mode = True
sampling_rate = 0.05
results_file = "/home/martin232009/buckets/b1/results.csv"
fieldnames = ["experiment_name", "seed", "training_time", "moving_average_rev"]
logging.info("comenzando")
features_to_drop = ["cprestamos_prendarios", "mprestamos_prendarios", "cprestamos_personales", "mprestamos_personales"]
canaritos = 10
gradient_bound = 0.01
n_seeds = 5
params = {
    "canaritos": canaritos,
    "gradient_bound": gradient_bound,
    "feature_fraction": 0.50,
    "is_unbalance": False,
}
# Leer datos
logger.info("Leyendo dataset")
df = pd.read_csv('~/datasets/competencia_02_target.csv')
if debug_mode:
    n_seeds = 1
    df = df.sample(frac=0.1)
    params["min_data_in_leaf"] = 2000
    params["gradient_bound"] = 0.4
    experiment_name += "_DEBUG"
    
experiment_name = f"{experiment_name}_c{canaritos}_gb{experiment_name}_s{sampling_rate}_u{(params['is_unbalance'])}"

# Eliminar features que no se van a usar
keep_cols = [col for col in df.columns if col not in features_to_drop]
df = df[keep_cols]
df = df[~df["foto_mes"].isna()]
# Agregar target y calcular weight
weight = {"BAJA+1": 1, "BAJA+2": 1.00002, "CONTINUA": 1}

# Preparar datos
start_time = time.time()
logger.info("Preparando datos")
X_train, y_train, w_train, X_eval, y_eval, w_eval, X_test, y_test = prepare_data(df, training_months, eval_month, test_month, get_features, weight, sampling_rate)
del df
gc.collect()

print_memory_state()

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

train_set = lgb.Dataset(X_train, label=y_train)
revs = train_models_and_save_results(train_set,X_eval, w_eval, params, seeds, results_file, save_model, n_seeds, experiment_name, fieldnames)
