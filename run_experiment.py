"""
Experimento delta-lags2: Evaluación de features con delta 2 y lag 2
"""
from datetime import datetime
import logging
import time

from dmeyf2025.experiments import experiment_init
from dmeyf2025.etl import ETL
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    experiment_config = experiment_init('config.yaml', script_file=__file__, debug=None)

    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"""\n{'=' * 70}
    📅 {date_time}
    📝 Iniciando experimento: {experiment_config['experiment_name']}
    🎯 Descripción: {experiment_config['config']['experiment']['description']}
    🔧 Experiment folder: {experiment_config['experiment_folder']}
{'=' * 70}""")

    start_time = time.time()

    #### ETL ####
    etl = ETL(experiment_config['raw_data_path'], [])
    etl.execute_complete_pipeline()

    logger.info(f"Experimento completado en {(time.time() - start_time)/60:.2f} minutos")

