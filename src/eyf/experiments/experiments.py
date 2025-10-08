"""
Módulo de gestión de experimentos
Proporciona funcionalidades para inicializar y configurar experimentos desde archivos YAML
"""

import yaml
from pathlib import Path


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


def experiment_init(config_path, debug=None, script_file=None):
    """
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
    
    # Determinar archivo del script para el nombre por defecto
    if script_file is None:
        # Obtener el archivo del caller (2 niveles arriba en el stack)
        import inspect
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back.f_back
            script_file = caller_frame.f_globals.get('__file__', 'unknown_script')
        finally:
            del frame
    
    # Validar y ajustar nombre del experimento
    experiment_name = config['experiment'].get('name')
    if not experiment_name or experiment_name.strip() == '':
        # Usar nombre del script si el nombre está vacío
        script_name = Path(script_file).stem
        config['experiment']['name'] = script_name
        experiment_name = script_name
        print(f"⚠️  Nombre de experimento vacío, usando nombre del script: {experiment_name}")
    
    # Sobrescribir modo debug si se proporciona
    if debug is not None:
        config['experiment']['debug'] = debug
        print(f"🔧 Modo debug sobrescrito a: {debug}")
    
    # Extraer variables de configuración
    DEBUG = config['experiment']['debug']
    SAMPLE_RATIO = config['experiment']['sample_ratio']
    n_trials = config['experiment']['n_trials']
    tag = config['experiment']['tag']
    experiment_folder = f"{experiment_name}_{tag}"
    
    # Ajustar configuración para modo debug
    if DEBUG:
        print(f"🚀 INICIANDO EXPERIMENTO {experiment_name} EN MODO DEBUG")
        n_trials = 2
        experiment_name = f"DEBUG_{experiment_name}"
        experiment_folder = f"DEBUG_{experiment_folder}"
    else:
        print(f"🚀 INICIANDO EXPERIMENTO {experiment_name}")
    
    # Configurar directorios (relativo al script que llama)
    script_dir = Path(script_file).parent
    if experiment_folder:
        experiment_dir = script_dir / experiment_folder
        experiment_dir.mkdir(parents=True, exist_ok=True)
    else:
        experiment_dir = script_dir
        
    # Reescribir archivo de configuración al directorio del experimento
    config_source = Path(config_path)
    config_destination = experiment_dir / config_source.name
    with open(config_destination, 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True, indent=2)
    print(f"📄 Archivo de configuración reescrito a: {config_destination}")
    
    # Configurar rutas de datos (3 niveles arriba desde el script)
    project_root = script_dir.parent.parent.parent
    data_path = project_root / 'data'
    raw_data_path = data_path / 'competencia_01_crudo.csv'
    target_data_path = data_path / 'competencia_01_target.csv'
    
    # Crear espacio de hiperparámetros
    hyperparameter_space = create_hyperparameter_space(config['hyperparameters'])
    
    # Retornar configuración completa
    return {
        'config': config,
        'DEBUG': DEBUG,
        'SAMPLE_RATIO': SAMPLE_RATIO,
        'n_trials': n_trials,
        'experiment_name': experiment_name,
        'experiment_folder': experiment_folder,
        'experiment_dir': experiment_dir,
        'hyperparameter_space': hyperparameter_space,
        'train_months': config['data']['train_months'],
        'test_month': config['data']['test_month'],
        'eval_month': config['data']['eval_month'],
        'n_lags': config['preprocessing']['lag_transformer']['n_lags'],
        'raw_data_path': raw_data_path,
        'target_data_path': target_data_path
    }


def validate_experiment_config(config):
    """
    Validar configuración de experimento
    
    Args:
        config (dict): Configuración del experimento
        
    Raises:
        ValueError: Si la configuración es inválida
    """
    required_sections = ['experiment', 'hyperparameters', 'data', 'preprocessing']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Sección requerida '{section}' no encontrada en configuración")
    
    # Validar sección experiment
    exp_config = config['experiment']
    required_exp_fields = ['tag', 'debug', 'sample_ratio', 'n_trials']
    for field in required_exp_fields:
        if field not in exp_config:
            raise ValueError(f"Campo requerido 'experiment.{field}' no encontrado")
    
    # Validar hiperparámetros
    for param_name, param_config in config['hyperparameters'].items():
        required_param_fields = ['type', 'min', 'max']
        for field in required_param_fields:
            if field not in param_config:
                raise ValueError(f"Campo requerido '{field}' no encontrado en hiperparámetro '{param_name}'")
    
    print("✅ Configuración de experimento validada correctamente")
