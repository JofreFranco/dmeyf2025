"""
Experimento Lags3: Evaluación de features con lag 1, 2 y 3
"""

import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
from time import time

from eyf.data.loading import load_and_prepare_data
from eyf.data.splitting import split_train_test_eval
from eyf.modeling.optimization import optimize_hyperparameters_with_optuna, save_trials
from eyf.utils.files import process_experiment_predictions
from eyf.data.preprocessing import LagTransformer
from eyf.utils.data_dict import RANDOM_SEEDS, EXCLUDE_COLS
from eyf.utils.replicator import replicate_experiment
from eyf.experiments import experiment_init


def main(config_path, debug=None):
    """Función principal del experimento"""
    
    # Inicializar configuración del experimento
    exp_config = experiment_init(config_path, debug, __file__)
    
    # Extraer variables de la configuración
    DEBUG = exp_config['DEBUG']
    SAMPLE_RATIO = exp_config['SAMPLE_RATIO']
    n_trials = exp_config['n_trials']
    experiment_name = exp_config['experiment_name']
    experiment_dir = exp_config['experiment_dir']
    hyperparameter_space = exp_config['hyperparameter_space']
    train_months = exp_config['train_months']
    test_month = exp_config['test_month']
    eval_month = exp_config['eval_month']
    n_lags = exp_config.get('n_lags', 3)  # Default para lags3
    raw_data_path = exp_config['raw_data_path']
    target_data_path = exp_config['target_data_path']
    
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_time = time()

    print("=" * 70)
    print(f"📅 {date_time}")
    print("=" * 70)

    # 2. Cargar y preprocesar datos
    print("📂 Cargando y preprocesando datos...")
    lag_transformer = LagTransformer(n_lags=n_lags)
    df = load_and_prepare_data(str(raw_data_path), str(target_data_path), lag_transformer)
    print(f"✅ Datos procesados: {df.shape[0]} filas, {df.shape[1]} columnas")

    # 3. Dividir datos y aplicar sampling
    print("✂️ Dividiendo datos y aplicando sampling...")

    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
    feature_names = feature_cols.copy()

    data_splits = split_train_test_eval(
        df, train_months, test_month, eval_month, 
        sample_ratio=SAMPLE_RATIO, debug_mode=DEBUG
    )

    # Extraer datos para optimización (con sampling)
    X_train, y_train, w_train = data_splits['train']
    X_test, y_test, w_test = data_splits['test']
    X_eval, customer_id_eval = data_splits['eval']

    if not DEBUG:
        # Extraer datos completos para modelo final (sin sampling)
        X_train_full, y_train_full, w_train_full = data_splits['train_full']
    else:
        X_train_full = None
        y_train_full = None
        w_train_full = None

    print(f"📊 Train: {len(X_train)} registros")
    print(f"📊 Test: {len(X_test)} registros") 
    print(f"📊 Eval: {len(X_eval)} registros")
    print(f"✅ Features: {X_train.shape[1]} columnas")

    # 4. Optimizar hiperparámetros con Optuna
    print(f"🔍 Iniciando optimización bayesiana...")
    study_name = f"{experiment_name}_optimization"

    study = optimize_hyperparameters_with_optuna(
        hyperparameter_space=hyperparameter_space,
        X_train=X_train,
        y_train=y_train, 
        w_train=w_train,
        n_trials=n_trials,
        seed=RANDOM_SEEDS[0],
        study_name=study_name,
        # feval=lgb_gan_eval  # Descomentar para optimizar sobre ganancia
    )

    # 6. Guardar hiperparámetros en JSON
    best_params = study.best_params
    json_filename = f"{experiment_name}.json"
    json_path = experiment_dir / json_filename
    with open(json_path, 'w') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    print(f"💾 Hiperparámetros guardados en: {json_filename}")
    save_trials(study, experiment_name, experiment_dir)
    gc.collect()

    # 5. Ejecutar replicación con todas las semillas
    print("\n🔄 Iniciando replicación con múltiples semillas...")
    print(f"📊 Para optimización: {len(X_train)} registros con sampling")

    results = replicate_experiment(
        X_train, y_train, w_train,  # Datos con sampling para entrenar modelos individuales
        X_test, y_test, w_test, 
        X_eval, customer_id_eval,
        working_dir=experiment_dir,
        experiment_name=experiment_name,
        feature_names=feature_names,
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        w_train_full=w_train_full
    )
    gc.collect()

    # 8. Mostrar resultados finales
    print("=" * 70)
    print(f"\n🎉 EXPERIMENTO COMPLETADO {experiment_name}! - {date_time}")
    print("=" * 70)
    print(f"📊 AUC promedio: {results['average_metrics']['auc_mean']:.4f} ± {results['average_metrics']['auc_std']:.4f}")
    print(f"💰 Ganancia promedio: {results['average_metrics']['gain_mean']:,.0f} ± {results['average_metrics']['gain_std']:,.0f}")
    print(f"🎯 Threshold optimizado: {results['average_metrics']['threshold_mean']:.4f} ± {results['average_metrics']['threshold_std']:.4f}")
    print(f"🏆 Ganancia con threshold óptimo: {results['average_metrics']['threshold_gain_mean']:,.0f} ± {results['average_metrics']['threshold_gain_std']:,.0f}")

    # 9. Generar archivos de predicciones binarias
    print("\n🎯 Generando predicciones binarias...")
    # Usar threshold optimizado del ensemble
    optimal_threshold = results.get('ensemble_threshold', 0.025)  # Fallback al threshold estándar
    print(f"🎯 Usando threshold optimizado: {optimal_threshold:.4f}")

    prediction_files = process_experiment_predictions(
        experiment_dir=experiment_dir,
        threshold=optimal_threshold,
        experiment_name=experiment_name
    )

    print("\n📁 Archivos generados")
    print(f"🕒 Tiempo de ejecución: {time() - start_time:.2f} segundos")
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ejecutar experimento Lags3 con configuración YAML')
    parser.add_argument('config_path', type=str, help='Ruta al archivo de configuración YAML')
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument('--debug', action='store_true', 
                             help='Forzar modo debug (sobrescribe YAML)')
    debug_group.add_argument('--no-debug', action='store_true',
                             help='Forzar modo no-debug (sobrescribe YAML)')
    args = parser.parse_args()
    
    # Determinar valor del debug
    debug_override = None
    if args.debug:
        debug_override = True
    elif args.no_debug:
        debug_override = False
    
    main(args.config_path, debug_override)