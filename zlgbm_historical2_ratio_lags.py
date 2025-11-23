import logging
import gc
import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from dmeyf2025.processors.feature_processors import CleanZerosTransformer, DeltaLagTransformer, PercentileTransformer, PeriodStatsTransformer, TendencyTransformer, IntraMonthTransformer, RandomForestFeaturesTransformer, DatesTransformer, HistoricalFeaturesTransformer, AddCanaritos, apply_transformer, AvgRatioTransformer, RatioLagsTransformer
from dmeyf2025.processors.sampler import SamplerProcessor
from dmeyf2025.utils.experiment_utils import *
from dmeyf2025.etl import prepare_data
from config import *

pd.set_option('display.max_columns', None)
orig_experiment_name = experiment_name
experiment_name = f'{orig_experiment_name}_c{canaritos}_gb{gradient_bound}_{sampler_conf["method"]}_{sampler_conf["target_sr"]}_p0{sampler_conf["p0"]}'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(experiment_log_file),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger(__name__)

def get_features(X, training_months):
    X_transformed = X
    X_transformed = apply_transformer(
        CleanZerosTransformer(),
        X_transformed,
        "CleanZerosTransformer"
    )
    initial_cols =X_transformed.columns
    X_transformed = apply_transformer(
        AvgRatioTransformer(months=3),
        X_transformed,
        "AvgRatioTransformer",
        parallel=True, parallelize_by='numero_de_cliente', n_jobs=-1
    )
    new_cols = list(set(X_transformed.columns) - set(initial_cols))
    X_transformed = apply_transformer(
        PeriodStatsTransformer(
            periods=[6],
            add_exclude_cols=new_cols
        ),
        X_transformed,
        "PeriodStatsTransformer",
        parallel=True, parallelize_by='numero_de_cliente', n_jobs=-1
    )
    new_cols = list(set(X_transformed.columns) - set(initial_cols))
    X_transformed = apply_transformer(
        TendencyTransformer(
            add_exclude_cols=new_cols
        ),
        X_transformed,
        "TendencyTransformer",
        parallel=True, parallelize_by='numero_de_cliente', n_jobs=-1
    )
    new_cols = list(set(X_transformed.columns) - set(initial_cols))
    X_transformed = apply_transformer(
        DeltaLagTransformer(
            n_lags=2,
            add_exclude_cols=new_cols
        ),
        X_transformed,
        "DeltaLagTransformer",
        parallel=True, parallelize_by='numero_de_cliente', n_jobs=-1
    )
    new_cols = list(set(X_transformed.columns) - set(initial_cols))
    X_transformed = apply_transformer(
        RatioLagsTransformer(
            n_lags=1,
            add_exclude_cols=new_cols
        ),
        X_transformed,
        "RatioLagsTransformer",
        parallel=True, parallelize_by='foto_mes', n_jobs=-1
    )
    X_transformed = apply_transformer(
        PercentileTransformer(
            replace_original=True
        ),
        X_transformed,
        "PercentileTransformer",
        parallel=True, parallelize_by='foto_mes', n_jobs=-1
    )
    return X_transformed


logger.info("=========>comenzando experimento %s", experiment_name)
logger.info("Leyendo dataset")
df = pd.read_csv(dataset_path)

if debug_mode:
    logger.info("Sampleando dataset modo debug")
    df = df.sample(frac=0.05)

# Eliminar features que no se van a usar
logger.info(f"Eliminando features que no se van a usar. La cantidad a eliminar es {len(features_to_drop)}")
keep_cols = [col for col in df.columns if col not in features_to_drop]
df = df[keep_cols]
df = df[~df["foto_mes"].isna()]
sampler_conf["target_sr"] = 1
start_time = time.time()
logger.info("Preparando datos")
X_train, y_train, w_train, X_eval, y_eval, w_eval, X_test, y_test = prepare_data(df, training_months, eval_month, test_month, get_features, weight, **sampler_conf)

gc.collect()
sampler_conf["target_sr"]=0.05
print_memory_state()
X_train["label"] = y_train
df_train = X_train.copy()
sampler = SamplerProcessor(**sampler_conf)
df_sampled = sampler.transform(df_train)
X_train, y_train = df_sampled.drop(columns=["label"]), df_sampled["label"]

logger.info("Agregando canaritos")
X_train = AddCanaritos(n_canaritos=canaritos).transform(X_train)
X_eval = AddCanaritos(n_canaritos=canaritos).transform(X_eval) 
logger.info("Datos pre procesados en tiempo: %s", time.time() - start_time)

train_set = lgb.Dataset(X_train, label=y_train)

revs = train_models_and_save_results(train_set,X_eval, w_eval, params, seeds, results_file, save_model, n_seeds, experiment_name, fieldnames, bucket_path, debug_mode)
#############################
#############################

sampler_conf["special_months"] = {201906: 0.1, 202006: 0.2}
experiment_name = f'{orig_experiment_name}_c{canaritos}_gb{gradient_bound}_{sampler_conf["method"]}_{sampler_conf["target_sr"]}_p0{sampler_conf["p0"]}_sp_0.1_0.2'
logger.info("=========>comenzando experimento %s", experiment_name)
start_time = time.time()
sampler = SamplerProcessor(**sampler_conf)
df_sampled = sampler.transform(df_train)
X_train, y_train = df_sampled.drop(columns=["label"]), df_sampled["label"]

gc.collect()

print_memory_state()

logger.info("Agregando canaritos")
X_train = AddCanaritos(n_canaritos=canaritos).transform(X_train)
X_eval = AddCanaritos(n_canaritos=canaritos).transform(X_eval) 
logger.info("Datos pre procesados en tiempo: %s", time.time() - start_time)

train_set = lgb.Dataset(X_train, label=y_train)

revs = train_models_and_save_results(train_set,X_eval, w_eval, params, seeds, results_file, save_model, n_seeds, experiment_name, fieldnames, bucket_path, debug_mode)


#############################
#############################

sampler_conf["method"] = "exponential"
sampler_conf["p0"] = 0.5
sampler_conf["special_months"] = None
experiment_name = f'{orig_experiment_name}_c{canaritos}_gb{gradient_bound}_{sampler_conf["method"]}_{sampler_conf["target_sr"]}_p0{sampler_conf["p0"]}'
logger.info("=========>comenzando experimento %s", experiment_name)
start_time = time.time()
logger.info("Preparando datos")
sampler = SamplerProcessor(**sampler_conf)
df_sampled = sampler.transform(df_train)
X_train, y_train = df_sampled.drop(columns=["label"]), df_sampled["label"]

gc.collect()

print_memory_state()

logger.info("Agregando canaritos")
X_train = AddCanaritos(n_canaritos=canaritos).transform(X_train)
X_eval = AddCanaritos(n_canaritos=canaritos).transform(X_eval) 
logger.info("Datos pre procesados en tiempo: %s", time.time() - start_time)

train_set = lgb.Dataset(X_train, label=y_train)

revs = train_models_and_save_results(train_set,X_eval, w_eval, params, seeds, results_file, save_model, n_seeds, experiment_name, fieldnames, bucket_path, debug_mode)

#############################
#############################

sampler_conf["method"] = "exponential"
sampler_conf["p0"] = 0.5
sampler_conf["special_months"] = {201906: 0.1, 202006: 0.2}
experiment_name = f'{orig_experiment_name}_c{canaritos}_gb{gradient_bound}_{sampler_conf["method"]}_{sampler_conf["target_sr"]}_p0{sampler_conf["p0"]}_sp_0.1_0.2'
logger.info("=========>comenzando experimento %s", experiment_name)
start_time = time.time()
sampler = SamplerProcessor(**sampler_conf)
df_sampled = sampler.transform(df_train)
X_train, y_train = df_sampled.drop(columns=["label"]), df_sampled["label"]

gc.collect()

print_memory_state()

logger.info("Agregando canaritos")
X_train = AddCanaritos(n_canaritos=canaritos).transform(X_train)
X_eval = AddCanaritos(n_canaritos=canaritos).transform(X_eval) 
logger.info("Datos pre procesados en tiempo: %s", time.time() - start_time)

train_set = lgb.Dataset(X_train, label=y_train)

revs = train_models_and_save_results(train_set,X_eval, w_eval, params, seeds, results_file, save_model, n_seeds, experiment_name, fieldnames, bucket_path, debug_mode)


#############################
#############################

sampler_conf["method"] = "exponential"
sampler_conf["p0"] = 0.3
sampler_conf["special_months"] = None
experiment_name = f'{orig_experiment_name}_c{canaritos}_gb{gradient_bound}_{sampler_conf["method"]}_{sampler_conf["target_sr"]}_p0{sampler_conf["p0"]}'
logger.info("=========>comenzando experimento %s", experiment_name)
start_time = time.time()
sampler = SamplerProcessor(**sampler_conf)
df_sampled = sampler.transform(df_train)
X_train, y_train = df_sampled.drop(columns=["label"]), df_sampled["label"]

gc.collect()

print_memory_state()

logger.info("Agregando canaritos")
X_train = AddCanaritos(n_canaritos=canaritos).transform(X_train)
X_eval = AddCanaritos(n_canaritos=canaritos).transform(X_eval) 
logger.info("Datos pre procesados en tiempo: %s", time.time() - start_time)

train_set = lgb.Dataset(X_train, label=y_train)

revs = train_models_and_save_results(train_set,X_eval, w_eval, params, seeds, results_file, save_model, n_seeds, experiment_name, fieldnames, bucket_path, debug_mode)