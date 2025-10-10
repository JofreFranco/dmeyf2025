"""
Experimento delta-lags2: Evaluación de features con delta 2 y lag 2
"""
import logging
from datetime import datetime
from dmeyf2025.experiments import experiment_init

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

    

    logger.info("Experimento completado")

