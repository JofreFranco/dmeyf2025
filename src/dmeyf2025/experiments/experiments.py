"""
Módulo de gestión de experimentos
Proporciona funcionalidades para inicializar y configurar experimentos desde archivos YAML
"""

import os
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def commit_experiment(experiment_dir, message):
    """Hacer un commit git del experimento"""
    import subprocess
    try:
        subprocess.run(
            ["git", "add", str(experiment_dir)],
            check=True
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            check=True
        )
        logger.info(f"✅ Commit realizado del experimento: {message}")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo realizar el commit git automático: {e}")


def load_config(config_path):
    """Cargar configuración desde archivo YAML"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def create_hyperparameter_space(hyperparams_config):
    """
    Crear espacio de hiperparámetros desde configuración YAML
    
    Args:
        hyperparams_config (dict): Configuración de hiperparámetros desde YAML
        
    Returns:
        dict: Espacio de hiperparámetros en formato (tipo, min, max)
    """
    hyperparameter_space = {}
    for param_name, param_config in hyperparams_config.items():
        param_type = param_config['type']
        param_min = param_config['min']
        param_max = param_config['max']
        hyperparameter_space[param_name] = (param_type, param_min, param_max)
    return hyperparameter_space

def write_config(config, experiment_dir):
    """Escribir configuración del experimento en el directorio del experimento"""
    with open(experiment_dir / "config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True, indent=2)

def write_logs(logs, experiment_dir):
    """Escribir logs del experimento en el directorio del experimento"""
    with open(experiment_dir / "logs.txt", "w", encoding="utf-8") as file:
        file.write(logs)

def setup_logging(experiment_dir):
    """Configurar logging a archivo en el directorio del experimento"""
    log_file = experiment_dir / "experiment.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    logging.getLogger().addHandler(file_handler)

def experiment_init(config_path, debug=False, script_file=None):
    """
    # TODO: Esto debería ser una clase
    Inicializar configuración del experimento desde archivo YAML
    
    Args:
        config_path (str): Ruta al archivo de configuración YAML
        debug (bool, optional): Sobrescribir modo debug del YAML
        script_file (str, optional): Ruta del script que ejecuta el experimento
        
    Returns:
        dict: Configuración completa del experimento
    """
    # Cargar configuración
    config = load_config(config_path)
    experiment_name = config['experiment'].get('name')
    if debug:
        config['experiment']['debug'] = debug
        config['experiment']['commit'] = False
        config['experiment']['sample_ratio'] = 0.001
        config['experiment']['n_trials'] = 2
        config['experiment']['n_init'] = 1
        config['experiment']['seeds'] = config['experiment']['seeds'][:1]
    
    # Extraer variables de configuración
    DEBUG = config['experiment']['debug']
    
    # Setear variable de entorno para debug
    os.environ['DEBUG_MODE'] = str(DEBUG)
    SAMPLE_RATIO = config['experiment']['sample_ratio']
    COMMIT = config['experiment']['commit']
    n_trials = config['experiment']['n_trials']
    n_init = config['experiment']['n_init']
    tag = config['experiment']['tag']
    version = config['experiment']['version']
    experiments_path = Path(config['experiment']['experiments_path'])
    experiment_folder = Path(f"{experiment_name}_{tag}_{version}")
    data_path = Path(config['experiment']['data_path'])
    raw_data_path = Path(config['experiment']['raw_data_path'])
    result_path = Path(config['experiment']['result_path'])
    positive_classes = config['experiment']['positive_classes']
    seeds = config['experiment']['seeds']
    if DEBUG:
        n_trials = 2
        experiment_name = f"DEBUG_{experiment_name}"
        experiment_folder = f"DEBUG_{experiment_folder}"

    experiment_dir = experiments_path / experiment_folder
    if DEBUG:
        print(experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
    else:
        experiment_dir.mkdir(parents=True, exist_ok=False)


    setup_logging(experiment_dir)
    
    if DEBUG:
        logger.info(f"🚀 INICIANDO EXPERIMENTO {experiment_name} EN MODO DEBUG")
    else:
        logger.info(f"🚀 INICIANDO EXPERIMENTO {experiment_name}")
    
    # Guardar configuración del experimento
    write_config(config, experiment_dir)
    
    # Hacer commit del experimento
    if config['experiment']['commit']:
        commit_experiment(experiment_dir, f"{experiment_name}_{tag}_{version}")

    hyperparameter_space = create_hyperparameter_space(config['hyperparameters'])
    weights = config.get('weights', {})
    
    return {
        'config': config,
        'DEBUG': DEBUG,
        'SAMPLE_RATIO': SAMPLE_RATIO,
        'n_trials': n_trials,
        'n_init': n_init,
        'version': version,
        'experiment_name': experiment_name,
        'experiment_folder': experiment_folder,
        'experiment_dir': experiment_dir,
        'hyperparameter_space': hyperparameter_space,
        'train_months': config['data']['train_months'],
        'test_month': config['data']['test_month'],
        'eval_month': config['data']['eval_month'],
        'weights': weights,
        'raw_data_path': raw_data_path,
        'result_path': result_path,
        'data_path': data_path,
        'experiments_path': experiments_path,
        'positive_classes': positive_classes,
        'seeds': seeds,
    }


