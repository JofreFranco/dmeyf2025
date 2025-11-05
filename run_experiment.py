"""
Experimento
"""
import argparse
from datetime import datetime
import logging
import time
import os
import random
import numpy as np
from dmeyf2025.experiments import experiment_init, save_experiment_results
from dmeyf2025.processors.feature_processors import CleanZerosTransformer, DeltaLagTransformer, PercentileTransformer, PeriodStatsTransformer, TendencyTransformer, IntraMonthTransformer, RandomForestFeaturesTransformer
from dmeyf2025.utils.features_check import check_features
from dmeyf2025.utils.data_dict import FINANCIAL_COLS
from dmeyf2025.utils.wilcoxon import compare_with_best_model
from dmeyf2025.utils.scale_params import scale_params
from dmeyf2025.pipelines import load_data, preprocessing_pipeline, optimization_pipeline, evaluation_pipeline, production_pipeline

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

def get_features(X, training_months):
    logger.info(f"Cantidad de features: {len(X.columns)}")
    initial_columns = set(X.columns)

    logger.info("Iniciando clean zeros transformer...")
    clean_zeros_transformer = CleanZerosTransformer(exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"])
    X_transformed = clean_zeros_transformer.fit_transform(X)
    logger.info("Iniciando intra month transformer...")
    intra_month_transformer = IntraMonthTransformer(exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"])
    X_transformed = intra_month_transformer.fit_transform(X_transformed)
    logger.info(f"Cantidad de features después de intra month transformer: {len(X_transformed.columns)}")

    logger.info("Iniciando tendency transformer...")
    tendency_transformer = TendencyTransformer(exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"])
    X_transformed = tendency_transformer.fit_transform(X_transformed)
    new_columns = set(X_transformed.columns) - initial_columns

    logger.info(f"Cantidad de features después de tendency transformer: {len(X_transformed.columns)}")

    logger.info("Iniciando period stats transformer...")
    period_stats_transformer = PeriodStatsTransformer(periods=[2, 3], exclude_cols=list(new_columns) + ["foto_mes", "numero_de_cliente", "target", "label", "weight"])
    X_transformed = period_stats_transformer.fit_transform(X_transformed)
    new_columns = set(X_transformed.columns) - initial_columns
    logger.info(f"Cantidad de features después de period stats transformer: {len(X_transformed.columns)}")

    logger.info("Iniciando delta lag transformer...")
    delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2, exclude_cols=list(new_columns) + ["foto_mes", "numero_de_cliente", "target", "label", "weight"])
    X_transformed = delta_lag_transformer.fit_transform(X_transformed)
    logger.info(f"Cantidad de features después de delta lag transformer: {len(X_transformed.columns)}")

    logger.info("Iniciando percentile transformer...")
    percentile_transformer = PercentileTransformer(variables=None, replace_original=True)
    X_transformed = percentile_transformer.fit_transform(X_transformed)
    logger.info(f"Cantidad de features después de percentile transformer: {len(X_transformed.columns)}")

    logger.info("Iniciando RandomForest Feature Transformer...")
    random_forest_features_transformer = RandomForestFeaturesTransformer(training_months= training_months)  
    X_transformed = random_forest_features_transformer.fit_transform(X_transformed)
    logger.info(f"Cantidad de features después de RandomForest Feature Transformer: {len(X_transformed.columns)}")
    return X_transformed

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
    X, y = load_data(experiment_config)
    
    # Preprocessing Pipeline
    X_train, y_train, w_train, X_eval, y_eval, w_eval, X_prod, y_prod, w_prod = preprocessing_pipeline(X, y, experiment_config, get_features)
    
    if False:
        check_features(X_train, 0.01) # Va a imprimir todo porque el umbral es bajo
        check_features(X_eval, 0.01)
        if X_prod is not None:
            check_features(X_prod, 0.01)
    
    # Optimization Pipeline
    best_params, X_train_sampled, y_train_sampled, w_train_sampled = optimization_pipeline(experiment_config, X_train, y_train, w_train, seeds)
  
    # Evaluation Pipeline
    experiment_path = experiment_config['experiment_dir']
    rev, n_sends = evaluation_pipeline(experiment_config, X_train, y_train, w_train, X_eval, y_eval, w_eval, best_params, seeds, experiment_path, DEBUG, X_train_sampled, y_train_sampled,w_train_sampled, is_hp_scaled=False)
    compare_with_best_model(rev, tracking_file=str(experiment_config['tracking_file_path']))
    save_experiment_results(experiment_config, rev, n_sends, np.mean(rev), np.median(rev), np.mean(n_sends), np.median(n_sends), hp_scaled=False)
    
    # Evaluation with HP Scaling Pipeline
    best_params_scaled = scale_params(best_params, X_train, X_train_sampled)
    rev_hp_scaled, n_sends_hp_scaled = evaluation_pipeline(experiment_config, X_train, y_train, w_train, X_eval, y_eval, w_eval, best_params_scaled, seeds, experiment_path, DEBUG, X_train_sampled, y_train_sampled,w_train_sampled, is_hp_scaled=True)
    compare_with_best_model(rev_hp_scaled, tracking_file=str(experiment_config['tracking_file_path']))
    save_experiment_results(experiment_config, rev_hp_scaled, n_sends_hp_scaled, np.mean(rev_hp_scaled), np.median(rev_hp_scaled), np.mean(n_sends_hp_scaled), np.median(n_sends_hp_scaled), hp_scaled=True)
    
    # Production Pipeline - Generar predicciones finales
    if experiment_config['config']['data'].get('production_month') and X_prod is not None:
        logger.info("\n" + "="*70)
        logger.info("Iniciando generación de predicciones de producción...")
        logger.info("="*70)
        
        # Guardar el mejor n_sends para usarlo en producción
        experiment_config['best_n_sends'] = int(np.median(n_sends))
        
        # Usar los mejores hiperparámetros escalados y datos ya procesados
        production_file = production_pipeline(
            experiment_config,
            X_train,
            y_train,
            w_train,
            X_eval,
            y_eval,
            w_eval,
            X_prod,
            best_params_scaled,  # Usar parámetros escalados
            seeds
        )
        logger.info(f"Archivo de producción generado: {production_file}")
    else:
        logger.info("No se generaron predicciones de producción (production_month no definido o sin datos)")
    
    logger.info(f"Experimento completado en {(time.time() - start_time)/60:.2f} minutos")

if __name__ == "__main__":
    main()

