"""
Experimento delta-lags2: Evaluación de features con delta 2 y lag 2
"""
from datetime import datetime
import logging
import time
import os
from sklearn.pipeline import Pipeline

from dmeyf2025.experiments import experiment_init
from dmeyf2025.etl import ETL
from dmeyf2025.processors.target_processor import BinaryTargetProcessor, CreateTargetProcessor
from dmeyf2025.processors.sampler import SamplerProcessor
from dmeyf2025.processors.feature_processors import DeltaLagTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


########## MAIN ##########
if __name__ == "__main__":
    experiment_config = experiment_init('config.yaml', script_file=__file__, debug=True)
    DEBUG = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"""\n{'=' * 70}
    📅 {date_time}
    📝 Iniciando experimento: {experiment_config['experiment_name']}
    🎯 Descripción: {experiment_config['config']['experiment']['description']}
    🔧 Experiment folder: {experiment_config['experiment_folder']}
{'=' * 70}""")

    start_time = time.time()
   
    #### ETL ####
    """
    Lee los datos, calcula target ternario, y divide en train, test y eval.
    """
    etl = ETL(experiment_config['raw_data_path'], CreateTargetProcessor(), experiment_config['config']['data']['train_months'], experiment_config['config']['data']['test_month'], experiment_config['config']['data']['eval_month'])
    X_train, y_train, X_test, y_test, X_eval, y_eval = etl.execute_complete_pipeline()
    
    #### TRAIN ####
    target_processor = BinaryTargetProcessor(experiment_config['config']['experiment']['positive_classes'])
    X_train, y_train = target_processor.fit_transform(X_train, y_train)
    sampler_processor = SamplerProcessor(experiment_config['SAMPLE_RATIO'])
    X_train_sampled, y_train_sampled = sampler_processor.fit_transform(X_train, y_train)

    #### Features ####
    delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2)
    X_train_transformed = delta_lag_transformer.fit_transform(X_train_sampled)
    X_test_transformed = delta_lag_transformer.transform(X_test)
    X_eval_transformed = delta_lag_transformer.transform(X_eval)

    logger.info(f"X_train_transformed.shape: {X_train_transformed.shape}")
    logger.info(f"X_test_transformed.shape: {X_test_transformed.shape}")
    logger.info(f"X_eval_transformed.shape: {X_eval_transformed.shape}")
    # Loggea todas las columnas, una por línea
    if DEBUG:
        logger.info("X_train_transformed.columns ({} columnas):\n{}".format(
            len(X_train_transformed.columns),
            "\n".join(X_train_transformed.columns)
        ))


    logger.info(f"Experimento completado en {(time.time() - start_time)/60:.2f} minutos")

