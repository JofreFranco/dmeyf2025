"""
Experimento
"""
import argparse
from datetime import datetime
import gc
import logging
import time
import os
import random
import numpy as np
import pandas as pd

from dmeyf2025.experiments import experiment_init, save_experiment_results
from dmeyf2025.etl import ETL
from dmeyf2025.processors.target_processor import BinaryTargetProcessor, CreateTargetProcessor
from dmeyf2025.processors.sampler import SamplerProcessor
from dmeyf2025.processors.feature_processors import DeltaLagTransformer, PercentileTransformer, PeriodStatsTransformer
from dmeyf2025.modelling.optimization import optimize_params
from dmeyf2025.modelling.train_model import train_models
from dmeyf2025.utils.data_dict import FINANCIAL_COLS
from dmeyf2025.metrics.revenue import sends_optimization
from dmeyf2025.pipelines import etl_pipeline, preprocessing_pipeline, optimization_pipeline, evaluation_pipeline #, evaluation_hp_scaling_pipeline
FORCE_DEBUG = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(
    description="Run experiment with specified config file."
    )
parser.add_argument(
    '--config', type=str, default='config.yaml', help='YAML config file to load'
    )
args = parser.parse_args()
CONFIG = args.config
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



def evaluation_hp_scaling_pipeline(experiment_config, X_train, y_train, X_eval, y_eval, best_params, seeds, experiment_path, DEBUG, X_train_sampled, y_train_sampled):
    """
    Evaluación con escalado de hiperparámetros
    """
    logger.info("Iniciando evaluación con escalado de hiperparámetros...")

    logger.info(f"min_data_in_leaf: {best_params['min_data_in_leaf']}")
    best_params["min_data_in_leaf"] = int(best_params["min_data_in_leaf"]/(len(X_train_sampled)/len(X_train)))
    logger.info(f"factor de escalado: {round(len(X_train_sampled)/len(X_train), 2)}")
    logger.info(f"min_data_in_leaf escalado: {best_params['min_data_in_leaf']}")
    
    if not DEBUG:
        X_final_train = X_train
        y_final_train = y_train
    else:
        X_final_train = X_train_sampled
        y_final_train = y_train_sampled
    
    predictions, models = train_models(X_final_train, y_final_train, X_eval, best_params, seeds, experiment_path)
    rev = []
    n_sends = []
    for model in models:
        y_pred = model.predict(X_eval)
        best_sends, max_rev = sends_optimization(y_pred, y_eval, min_sends=8000, max_sends=13000)
        rev.append(max_rev)
        n_sends.append(best_sends)

    y_pred = predictions["pred_ensemble"]
    best_sends, max_rev = sends_optimization(y_pred, y_eval, min_sends=8000, max_sends=13000)
    rev.append(max_rev)
    n_sends.append(best_sends)

    logger.info(f"N_sends HP Scaled Median:: {np.median(n_sends)}")
    logger.info(f"N_sends HP Scaled mean:: {np.mean(n_sends)}")
    logger.info(f"N_sends HP Scaled: {n_sends}")

    logger.info(f"rev HP Scaled Median:: {np.median(rev)}")
    logger.info(f"rev HP Scaled mean:: {np.mean(rev)}")
    logger.info(f"rev HP Scaled: {rev}")
    
    return rev, n_sends

def main():
    """
    Función principal que ejecuta todo el pipeline del experimento
    """
    # Inicializar experimento
    experiment_config = experiment_init(CONFIG, script_file=__file__, debug=FORCE_DEBUG)

    DEBUG = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    seeds = experiment_config["seeds"]

    np.random.seed(seeds[0])
    random.seed(seeds[0])

    # Logging inicial
    logger.info(
        f"""\n{'=' * 70}
    📅 {date_time}
    📝 Iniciando experimento: {experiment_config['experiment_name']}
    🎯 Descripción: {experiment_config['config']['experiment']['description']}
    🔧 Experiment folder: {experiment_config['experiment_folder']}
{'=' * 70}"""
    )
    start_time = time.time()
    
    # ETL Pipeline
    X, y = etl_pipeline(experiment_config)
    
    # Preprocessing Pipeline
    X_train, y_train, X_eval, y_eval = preprocessing_pipeline(X, y, experiment_config, get_features)
    
    # Optimization Pipeline
    best_params, X_train_sampled, y_train_sampled = optimization_pipeline(experiment_config, X_train, y_train, seeds)
    
    # Evaluation Pipeline
    experiment_path = f"{experiment_config['experiments_path']}/{experiment_config['experiment_folder']}"
    rev, n_sends = evaluation_pipeline(experiment_config, X_train, y_train, X_eval, y_eval, best_params, seeds, experiment_path, DEBUG, X_train_sampled, y_train_sampled)
    
    # Evaluation with HP Scaling Pipeline
    rev_hp_scaled, n_sends_hp_scaled = evaluation_hp_scaling_pipeline(experiment_config, X_train, y_train, X_eval, y_eval, best_params, seeds, experiment_path, DEBUG, X_train_sampled, y_train_sampled)
    
    # Guardar resultados finales con ambos modelos
    save_experiment_results(experiment_config, rev, n_sends, np.mean(rev), np.median(rev), np.mean(n_sends), np.median(n_sends), np.mean(rev_hp_scaled), np.mean(n_sends_hp_scaled))
    
    logger.info(f"Experimento completado en {(time.time() - start_time)/60:.2f} minutos")


if __name__ == "__main__":
    main()

