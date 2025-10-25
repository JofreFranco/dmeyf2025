import logging
import gc
import numpy as np
import pandas as pd
from dmeyf2025.etl import ETL
from dmeyf2025.processors.target_processor import BinaryTargetProcessor, CreateTargetProcessor
from dmeyf2025.processors.sampler import SamplerProcessor
from dmeyf2025.processors.feature_processors import DeltaLagTransformer, PercentileTransformer, PeriodStatsTransformer
from dmeyf2025.modelling.optimization import optimize_params
from dmeyf2025.modelling.train_model import train_models
from dmeyf2025.metrics.revenue import sends_optimization
from dmeyf2025.experiments import save_experiment_results
logger = logging.getLogger(__name__)

def etl_pipeline(experiment_config):
    """
    Lee los datos, calcula target ternario, y divide en train, test y eval.
    """
    logger.info("Iniciando ETL pipeline...")
    etl = ETL(experiment_config['raw_data_path'], CreateTargetProcessor(), 
              train_months=[202101, 202102, 202103, 202104, 202105, 202106])
    X, y, _, _, _, _ = etl.execute_complete_pipeline()
    if experiment_config['DEBUG']:
        X = X.sample(frac=0.1, random_state=42)
        y = y.sample(frac=0.1, random_state=42)
    return X, y

def preprocessing_pipeline(X, y, experiment_config, get_features):
    """
    Procesamiento de target y features
    """
    logger.info("Iniciando preprocessing pipeline...")
    
    # Target Processing
    target_processor = BinaryTargetProcessor(experiment_config['config']['experiment']['positive_classes'])
    X, y = target_processor.fit_transform(X, y)
    X["label"] = y
    
    # Features Processing
    logger.info("Iniciando procesamiento de features...")
    logger.debug(f"X.shape: {X.shape} - Número de cliente está incluido")

    X_transformed = get_features(X)
    X_transformed.set_index("numero_de_cliente", inplace=True)
    logger.debug(f"X_transformed.shape: {X_transformed.shape} - Sin número de cliente")
    
    # Split Data
    logger.info("Iniciando split de datos...")
    X_train = X_transformed[X_transformed["foto_mes"].isin(experiment_config['train_months'])]
    y_train = X_train["label"]
    X_train = X_train.drop(columns=["label"])
    logger.info(f"X_train.shape: {X_train.shape}")

    X_eval = X_transformed[X_transformed["foto_mes"].isin([experiment_config['eval_month']])]
    y_eval = X_eval["label"]
    X_eval = X_eval.drop(columns=["label"])
    logger.info(f"X_eval.shape: {X_eval.shape}")
    
    return X_train, y_train, X_eval, y_eval

def optimization_pipeline(experiment_config, X_train, y_train, seeds):
    """
    Optimización de hiperparámetros
    """
    logger.info("Iniciando optimización...")

    logger.info("Iniciando muestreo de datos...")

    sampler_processor = SamplerProcessor(experiment_config['SAMPLE_RATIO'], random_state=seeds[0])
    X_train_sampled, y_train_sampled = sampler_processor.fit_transform(X_train, y_train)
    logger.info(f"X_train_sampled.shape: {X_train_sampled.shape}")
    logger.info(f"y_train_sampled.shape: {y_train_sampled.shape}")
    
    best_params, _ = optimize_params(experiment_config, X_train_sampled, y_train_sampled, seed=seeds[0])
    gc.collect()
    
    return best_params, X_train_sampled, y_train_sampled

def evaluation_pipeline(experiment_config, X_train, y_train, X_eval, y_eval, best_params, seeds, experiment_path, DEBUG, X_train_sampled, y_train_sampled, is_hp_scaled=False):
    """
    Evaluación con modelos finales
    """
    logger.info("Iniciando Evaluación...")

    if not DEBUG:
        X_final_train = X_train
        y_final_train = y_train
    else:
        X_final_train = X_train_sampled
        y_final_train = y_train_sampled

    logger.debug(f"X_train.shape final training: {X_final_train.shape}")
    logger.debug(f"X_eval.shape final evaluation: {X_eval.shape}")

    n_seeds = len(seeds)
    logger.info(f"Entrenando y prediciendo con {n_seeds} seeds para ensamblado...")
    
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

    if is_hp_scaled:
        logger.info(f"N_sends HP Scaled Median:: {np.median(n_sends)}")
        logger.info(f"N_sends HP Scaled mean:: {np.mean(n_sends)}")
        logger.info(f"N_sends HP Scaled: {n_sends}")
    else:
        logger.info(f"N_sends Median:: {np.median(n_sends)}")
        logger.info(f"N_sends mean:: {np.mean(n_sends)}")
        logger.info(f"N_sends: {n_sends}")

    if is_hp_scaled:
        logger.info(f"rev HP Scaled Median:: {np.median(rev)}")
        logger.info(f"rev HP Scaled mean:: {np.mean(rev)}")
        logger.info(f"rev HP Scaled: {rev}")
    else:
        logger.info(f"rev Median:: {np.median(rev)}")
        logger.info(f"rev mean:: {np.mean(rev)}")
        logger.info(f"rev: {rev}")
    if is_hp_scaled:
        experiment_config['experiment_name'] = f"{experiment_config['experiment_name']} HP Scaled"
    # Guardar resultados del primer modelo
    
    
    return rev, n_sends
