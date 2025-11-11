from contextlib import ExitStack
import os
import logging
import gc
import numpy as np
import pandas as pd
from dmeyf2025.etl import ETL
from dmeyf2025.processors.target_processor import BinaryTargetProcessor, CreateTargetProcessor
from dmeyf2025.processors.sampler import SamplerProcessor
from dmeyf2025.modelling.optimization import optimize_params
from dmeyf2025.modelling.train_model import train_models, prob_to_sends
from dmeyf2025.metrics.revenue import sends_optimization
from dmeyf2025.utils.feature_importance import save_feature_importance_from_models

logger = logging.getLogger(__name__)

def load_data(experiment_config):
    """
    Lee todos los datos y calcula target ternario.
    """
    logger.info("Iniciando ETL pipeline...")
    etl = ETL(experiment_config['raw_data_path'], 
              CreateTargetProcessor(str(experiment_config['target_path'])), 
              blacklist_features=experiment_config['blacklist_features'],
              hard_filter=experiment_config['hard_filter'])
    X, y = etl.execute_complete_pipeline()
    if experiment_config['DEBUG']:
        X = X.sample(frac=0.1, random_state=42)
        y = y.sample(frac=0.1, random_state=42)
    return X, y

def preprocessing_pipeline(X, y, experiment_config, get_features, drop_features_csv=None):
    """
    Procesamiento de target y features
    """
    logger.info("Iniciando preprocessing pipeline...")
    
    # Target Processing
    target_processor = BinaryTargetProcessor(experiment_config['config']['experiment']['positive_classes'])
    X, y, y_weight = target_processor.fit_transform(X, y)
    X["label"] = y
    X["weight"] = y_weight
    # Features Processing
    logger.info("Iniciando procesamiento de features...")
    logger.debug(f"X.shape: {X.shape} - Número de cliente está incluido")

    X_transformed = get_features(X, experiment_config['train_months'])
    logger.debug(f"X_transformed.shape: {X_transformed.shape} - Con número de cliente como columna")
    if experiment_config['config']['experiment']['debug']:
        columns_output_path = experiment_config.get('columns_output_path', 'all_columns_X_transformed.txt')
        with open(columns_output_path, 'w') as f:
            for col in X_transformed.columns:
                f.write(f"{col}\n")
        logger.info(f"Nombres de columnas de X_transformed guardados en {columns_output_path}")
    # Eliminar features del CSV (si existe)
    if drop_features_csv:
        if os.path.exists(drop_features_csv):
            logger.info(f"Eliminando features especificadas en {drop_features_csv}...")
            df_features = pd.read_csv(drop_features_csv)
            if 'feature' in df_features.columns:
                features_to_drop = df_features['feature'].dropna().unique().tolist()
                features_to_drop_existing = [f for f in features_to_drop if f in X_transformed.columns]
                if features_to_drop_existing:
                    X_transformed = X_transformed.drop(columns=features_to_drop_existing)
                    logger.info(f"✅ Eliminadas {len(features_to_drop_existing)} features")
            else:
                logger.warning(f"El CSV {drop_features_csv} no tiene columna 'feature'")
        else:
            logger.warning(f"El archivo {drop_features_csv} no existe")
    
    logger.info(f"Features finales antes de split: {X_transformed.shape[1]}")
    
    # Split Data
    logger.info("Iniciando split de datos...")
    X_train = X_transformed[X_transformed["foto_mes"].isin(experiment_config['train_months'])].copy()
    y_train = X_train["label"]
    w_train = X_train["weight"]
    X_train = X_train.drop(columns=["label", "weight"])
    logger.info(f"X_train.shape: {X_train.shape}")

    X_eval = X_transformed[X_transformed["foto_mes"].isin([experiment_config['eval_month']])].copy()

    y_eval = X_eval["label"]
    w_eval = X_eval["weight"]
    X_eval = X_eval.drop(columns=["label", "weight"])
    logger.info(f"X_eval.shape: {X_eval.shape}")
    
    # Production Data (si existe)
    production_month = experiment_config['config']['data'].get('production_month')
    if production_month:
        X_prod = X_transformed[X_transformed["foto_mes"] == production_month].copy()
        if not X_prod.empty:
            y_prod = X_prod["label"]  # Dummy label
            w_prod = X_prod["weight"]  # Dummy weight
            X_prod = X_prod.drop(columns=["label", "weight"])
            logger.info(f"X_prod.shape: {X_prod.shape}")
        else:
            X_prod, y_prod, w_prod, prod_clientes = None, None, None, None
            logger.warning(f"No hay datos para production_month: {production_month}")
    else:
        X_prod, y_prod, w_prod = None, None, None
    
    return X_train, y_train, w_train, X_eval, y_eval, w_eval, X_prod, y_prod, w_prod

def optimization_pipeline(experiment_config, X_train, y_train, w_train, seeds):
    """
    Optimización de hiperparámetros
    """
    logger.info("Iniciando optimización...")

    logger.info("Iniciando muestreo de datos...")
    X_train["weight"] = w_train
    X_train = X_train.copy() # TODO: Esto es para evitar warnings de pandas, buscar algo mejor tal vez.
    sampler_processor = SamplerProcessor(experiment_config['SAMPLE_RATIO'], random_state=seeds[0])
    X_train_sampled, y_train_sampled = sampler_processor.fit_transform(X_train, y_train)
    W_train_sampled = X_train_sampled["weight"]
    X_train_sampled = X_train_sampled.drop(columns=["weight"])
    logger.info(f"X_train_sampled.shape: {X_train_sampled.shape}")
    logger.info(f"y_train_sampled.shape: {y_train_sampled.shape}")
    best_params, _ = optimize_params(experiment_config, X_train_sampled, y_train_sampled, W_train_sampled, seed=seeds[0])
    gc.collect()
    
    return best_params, X_train_sampled, y_train_sampled, W_train_sampled

def evaluation_pipeline(experiment_config, X_train, y_train, w_train, X_eval, y_eval, w_eval, best_params, seeds, experiment_path, DEBUG, X_train_sampled, y_train_sampled,w_train_sampled, is_hp_scaled=False):
    """
    Evaluación con modelos finales
    """
    logger.info("Iniciando Evaluación...")

    if not DEBUG:
        X_final_train = X_train
        y_final_train = y_train
        w_final_train = w_train
    else:
        X_final_train = X_train_sampled
        y_final_train = y_train_sampled
        w_final_train = w_train_sampled
    logger.debug(f"X_train.shape final training: {X_final_train.shape}")
    logger.debug(f"X_eval.shape final evaluation: {X_eval.shape}")

    n_seeds = len(seeds)
    logger.info(f"Entrenando y prediciendo con {n_seeds} seeds para ensamblado...")
    
    predictions, models = train_models(X_final_train, y_final_train, X_eval, best_params, seeds, w_final_train, experiment_path)
    

    if not is_hp_scaled:
        save_feature_importance_from_models(models, experiment_path, top_n=30)
    
    rev = []
    n_sends = []
    predictions = predictions.drop(columns=["numero_de_cliente"])
    for i in range(len(predictions.columns)-1):
        best_sends, max_rev = sends_optimization(predictions[f"pred_{i}"], w_eval)
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

def production_pipeline(experiment_config, X_train, y_train, w_train, X_eval, y_eval, w_eval, X_prod, best_params, seeds):
    """
    Pipeline de producción que genera predicciones finales para el mes de producción.
    Usa los datos ya procesados del preprocessing_pipeline.
    
    Args:
        experiment_config: Configuración del experimento
        X_train: Datos de entrenamiento procesados
        y_train: Target de entrenamiento
        w_train: Weights de entrenamiento
        X_eval: Datos de evaluación procesados
        y_eval: Target de evaluación
        w_eval: Weights de evaluación
        X_prod: Datos de producción procesados
        best_params: Mejores hiperparámetros encontrados
        seeds: Lista de seeds para el ensemble
    
    Returns:
        str: Path del archivo de predicciones guardado
    """
    logger.info("="*70)
    logger.info("Iniciando Production Pipeline...")
    logger.info("="*70)
    
    if X_prod is None or X_prod.empty:
        raise ValueError("No hay datos de producción disponibles")
    
    production_month = experiment_config['config']['data'].get('production_month')
    logger.info(f"Mes de producción: {production_month}")
    
    # Juntar datos de train y eval para entrenar el modelo final
    logger.info("Juntando datos de train y eval para entrenamiento final...")
    
    X_full = pd.concat([X_train, X_eval], axis=0)
    y_full = pd.concat([y_train, y_eval], axis=0)
    w_full = pd.concat([w_train, w_eval], axis=0)
    
    logger.info(f"Datos de entrenamiento final: {X_full.shape[0]} filas")
    logger.info(f"Datos de producción: {X_prod.shape[0]} filas")
    logger.info(f"Features: {X_full.shape[1]}")
    
    # Entrenar modelos con todos los seeds
    logger.info(f"Entrenando modelos de producción con {len(seeds)} seeds...")
    experiment_path = experiment_config['experiment_dir']
    predictions, models = train_models(
        X_full, 
        y_full, 
        X_prod, 
        best_params, 
        seeds, 
        w_full, 
        experiment_path
    )
    
    # Usar el número óptimo de envíos del experimento
    n_sends = experiment_config.get('best_n_sends', 11000)
    logger.info(f"Generando predicciones para {n_sends} envíos")
    
    # Guardar predicciones
    output_file = prob_to_sends(experiment_config, predictions, n_sends, name="production")
    
    logger.info("="*70)
    logger.info(f"✅ Predicciones de producción guardadas en: {output_file}")
    logger.info(f"   - Total clientes: {len(predictions)}")
    logger.info(f"   - Predicciones positivas: {n_sends}")
    logger.info("="*70)
    
    return output_file

