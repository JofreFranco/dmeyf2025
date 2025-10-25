"""
Experimento
"""
from datetime import datetime
import gc
import logging
import time
import os
import random
import numpy as np
import pandas as pd

from dmeyf2025.experiments import experiment_init
from dmeyf2025.etl import ETL
from dmeyf2025.processors.target_processor import BinaryTargetProcessor, CreateTargetProcessor
from dmeyf2025.processors.sampler import SamplerProcessor
from dmeyf2025.processors.feature_processors import DeltaLagTransformer, PercentileTransformer, PeriodStatsTransformer
from dmeyf2025.modelling.optimization import optimize_params
from dmeyf2025.modelling.train_model import train_models, prob_to_sends
from dmeyf2025.utils.data_dict import FINANCIAL_COLS
from dmeyf2025.metrics.revenue import sends_optimization
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def get_features(X):
    logger.info("Iniciando period stats transformer...")
    period_stats_transformer = PeriodStatsTransformer(period=2)
    X_transformed = period_stats_transformer.fit_transform(X)
    delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2)
    logger.info("Iniciando delta lag transformer...")
    X_transformed = delta_lag_transformer.fit_transform(X_transformed)
    percentile_transformer = PercentileTransformer(variables=FINANCIAL_COLS)
    logger.info("Iniciando percentile transformer...")
    X_transformed = percentile_transformer.fit_transform(X_transformed)
    return X_transformed

########## MAIN ##########
if __name__ == "__main__":
    import argparse
    #### CONFIG ####
    force_debug = True
    parser = argparse.ArgumentParser(
        description="Run experiment with specified config file."
        )
    parser.add_argument(
        '--config', type=str, default='config.yaml', help='YAML config file to load'
        )
    args = parser.parse_args()
    config_file = args.config

    #### INIT EXPERIMENT ####
    experiment_config = experiment_init(config_file, script_file=__file__, debug=force_debug)

    DEBUG = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    seeds = experiment_config["seeds"]
    experiment_path = f"{experiment_config['experiments_path']}/{experiment_config['experiment_folder']}"
    n_sends = experiment_config["n_sends"]

    np.random.seed(seeds[0])
    random.seed(seeds[0])
    
    #########################################################
    ###################INICIO EXPERIMENTO###################
    logger.info(
        f"""\n{'=' * 70}
    📅 {date_time}
    📝 Iniciando experimento: {experiment_config['experiment_name']}
    🎯 Descripción: {experiment_config['config']['experiment']['description']}
    🔧 Experiment folder: {experiment_config['experiment_folder']}
{'=' * 70}"""
)
    start_time = time.time()

    #### 1 - ETL ####
    """
    Lee los datos, calcula target ternario, y divide en train, test y eval.
    """
    etl = ETL(experiment_config['raw_data_path'], CreateTargetProcessor(), train_months = [202101, 202102, 202103, 202104, 202105, 202106],)
    X, y, _,_,_,_ = etl.execute_complete_pipeline()
    
    
    #### TARGET PROCESSING ####
    target_processor = BinaryTargetProcessor(experiment_config['config']['experiment']['positive_classes'])
    X, y = target_processor.fit_transform(X, y)
    X["label"] = y

    #### FEATURES PROCESSING ####
    logger.info("Iniciando procesamiento de features...")
    logger.info(f"X.shape: {X.shape} - Número de cliente está incluido")

    X_transformed = get_features(X)
    X_transformed.set_index("numero_de_cliente", inplace=True)
    logger.info(f"X_transformed.shape: {X_transformed.shape} - Sin número de cliente")
    
    #### SPLIT DATA ####

    logger.info("Iniciando split de datos...")
    X_train = X_transformed[X_transformed["foto_mes"].isin(experiment_config['train_months'])]
    y_train = X_train["label"]
    X_train = X_train.drop(columns=["label"])
    logger.info(f"X_train.shape: {X_train.shape}")

    X_eval = X_transformed[X_transformed["foto_mes"].isin([experiment_config['eval_month']])]
    y_eval = X_eval["label"]
    X_eval = X_eval.drop(columns=["label"])
    logger.info(f"X_eval.shape: {X_eval.shape}")


    #### 2 - OPTIMIZACIÓN DE HIPERPARÁMETROS ################## ###########################################################

    logger.info("Iniciando optimización...")

    logger.info("Iniciando muestreo de datos...")
    sampler_processor = SamplerProcessor(experiment_config['SAMPLE_RATIO'], random_state=seeds[0])
    X_train_sampled, y_train_sampled = sampler_processor.fit_transform(X_train, y_train)
    logger.info(f"X_train_sampled.shape: {X_train_sampled.shape}")
    logger.info(f"y_train_sampled.shape: {y_train_sampled.shape}")
    # TODO: Acá test de consistencia en el target
    best_params, _ = optimize_params(experiment_config, X_train_sampled, y_train_sampled, seed = seeds[0])
    gc.collect()

    #### Final Modeling #######################################
    ###########################################################

    logger.info("Iniciando Modelado final...")

    if not DEBUG:
        X_final_train = X_train
        y_final_train = y_train
    else:
        X_final_train = X_train_sampled
        y_final_train = y_train_sampled

    
    logger.info(f"X_train.shape final training: {X_final_train.shape}")
    n_seeds = len(seeds)
    logger.info(f"Entrenando y prediciendo con {n_seeds} seeds para ensamblado...")
    
    predictions, _ = train_models(X_final_train, y_final_train, X_eval, best_params, seeds, experiment_path)
    y_pred = predictions["pred_ensemble"]
    n_sends, max_ganancia = sends_optimization(y_pred, y_eval, min_sends=8000, max_sends=13000)
    logger.info(f"N_sends: {n_sends}")
    logger.info(f"Gain: {max_ganancia}")
    #X["label"] = y
    

    #### Final Modeling Con Escalado de Hiperparámetros ####
    ########################################################
    logger.info("Iniciando Modelado final con escalado de hiperparámetros...")

    logger.info(f"min_data_in_leaf: {best_params['min_data_in_leaf']}")
    best_params["min_data_in_leaf"] = int(best_params["min_data_in_leaf"]/(len(X_train_sampled)/len(X_final_train)))
    logger.info(f"factor de escalado: {round(len(X_train_sampled)/len(X_final_train), 2)}")
    logger.info(f"min_data_in_leaf escalado: {best_params['min_data_in_leaf']}")
    
    predictions, _ = train_models(X_final_train, y_final_train, X_eval, best_params, seeds, experiment_path)

    y_pred = predictions["pred_ensemble"]
    n_sends, max_ganancia = sends_optimization(y_pred, y_eval, min_sends=8000, max_sends=13000)
    logger.info(f"N_sends: {n_sends}")
    logger.info(f"Gain: {max_ganancia}")

    
    logger.info(f"Experimento completado en {(time.time() - start_time)/60:.2f} minutos")

