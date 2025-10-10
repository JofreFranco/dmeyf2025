"""
Experimento delta-lags2: Evaluación de features con delta 2 y lag 2
"""
import logging
from dmeyf2025.experiments import experiment_init

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Iniciando experimento delta-lags2")
    experiment_init('config.yaml', script_file=__file__, debug=None)
    logger.info("Experimento completado")

