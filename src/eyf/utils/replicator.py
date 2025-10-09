import os
import json
import gc

import pandas as pd
import numpy as np
import lightgbm as lgb

from eyf.modeling.callbacks import lgb_gan_eval
from eyf.modeling.metrics import ganancia_prob, calcular_auc
from eyf.modeling.optimization import optimize_threshold, detect_hardware_capabilities
from eyf.utils.plots import plot_feature_importance
from eyf.utils.data_dict import RANDOM_SEEDS
from eyf.utils.files import convertir_a_nativo


def get_optimized_lgb_params(base_params=None, quiet=True):
    """
    Obtiene parámetros optimizados de LightGBM basados en hardware disponible.
    
    Parameters:
    -----------
    base_params : dict, optional
        Parámetros base a optimizar
    quiet : bool, default=True
        Si True, no muestra mensajes de detección de hardware
        
    Returns:
    --------
    dict
        Parámetros optimizados para LightGBM
    """
    if base_params is None:
        base_params = {}
    
    # Detectar hardware (silenciosamente si quiet=True)
    if not quiet:
        hw_capabilities = detect_hardware_capabilities()
    else:
        hw_capabilities = {
            'num_cores': os.cpu_count(),
            'has_gpu': False,
            'device_type': 'cpu'
        }
        
        # Detección silenciosa de GPU
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            if platforms:
                hw_capabilities['has_gpu'] = True
                hw_capabilities['device_type'] = 'gpu'
        except:
            pass
    
    # Configurar parámetros optimizados
    optimized_params = base_params.copy()
    optimized_params.update({
        'device_type': hw_capabilities['device_type'],
        'num_threads': -1,                    # Usar todos los cores
        'force_row_wise': True,              # Optimización de memoria
        'max_bin': 255,                      # Mejor precisión
        'bin_construct_sample_cnt': 200000,  # Histogramas optimizados
        'data_sample_strategy': 'bagging',   # Sampling optimizado
        'feature_pre_filter': False,         # No filtrar features
        'max_cat_threshold': 32,             # Categorías automáticas
        'cat_smooth': 10                     # Smoothing categóricas
    })
    
    # Parámetros adicionales para GPU
    if hw_capabilities['has_gpu']:
        optimized_params.update({
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'min_data_in_leaf': max(20, optimized_params.get('min_data_in_leaf', 20))
        })
    
    return optimized_params

def load_hyperparameters(experiment_dir, experiment_name):
    """
    Carga los hiperparámetros desde un archivo JSON.

    Parameters:
    -----------
    experiment_dir : str
        Directorio donde está el experimento
    experiment_name : str
        Nombre del experimento sin extensión

    Returns:
    --------
    dict
        Diccionario con hiperparámetros
    """
    
    json_filename = f"{experiment_name}.json"
    json_path = os.path.join(experiment_dir, json_filename)
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No se encontró el archivo de hiperparámetros: {json_path}")
    
    with open(json_path, 'r') as f:
        params = json.load(f)
    
    print(f"✅ Hiperparámetros cargados desde: {json_path}")
    return params


def train_model_single_seed(X_train, y_train, w_train, X_test, y_test, w_test, 
                           params, seed, verbose=False):
    """
    Entrena un modelo LightGBM con una semilla específica.

    Parameters:
    -----------
    X_train, X_test : array-like
        Features de entrenamiento y test
    y_train, y_test : array-like
        Targets de entrenamiento y test
    w_train, w_test : array-like
        Pesos de entrenamiento y test
    params : dict
        Hiperparámetros del modelo
    seed : int
        Semilla aleatoria
    verbose : bool
        Si mostrar logs del entrenamiento

    Returns:
    --------
    tuple
        (modelo, predicciones_test, auc, ganancia)
    """
    # Obtener parámetros optimizados para hardware
    base_params = params.copy()
    base_params.update({
        'seed': seed,
        'verbose': 0 if not verbose else 1
    })
    
    params_with_seed = get_optimized_lgb_params(base_params, quiet=True)

    train_dataset = lgb.Dataset(X_train, label=y_train, weight=w_train)
    test_dataset = lgb.Dataset(X_test, label=y_test, reference=train_dataset, weight=w_test)

    # Extraer parámetros especiales que no van directamente a LightGBM
    params_copy = params_with_seed.copy()
    early_stopping_rounds = params_copy.pop('early_stopping_rounds', 50)
    num_boost_round = params_copy.pop('num_boost_round', 1000)

    modelo = lgb.train(
        params_copy,
        train_dataset,
        num_boost_round=num_boost_round,
        valid_sets=[test_dataset],
        valid_names=['test'],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(period=0)
        ],
        feval=lgb_gan_eval,
    )

    y_pred_test = modelo.predict(X_test)

    auc = calcular_auc(y_test, y_pred_test)
    # Convertir etiquetas binarias a clase ternaria para ganancia_prob
    # Solo BAJA+2 es considerada positiva
    y_ternaria = np.where(y_test == 1, "BAJA+2", "CONTINUA")
    # Crear array 2D para ganancia_prob (columna 0: prob negativa, columna 1: prob positiva)
    y_hat_2d = np.column_stack([1 - y_pred_test, y_pred_test])
    
    ganancia = ganancia_prob(y_hat_2d, y_ternaria, prop=1, class_index=1, threshold=0.025)

    return modelo, y_pred_test, auc, ganancia


def replicate_experiment(X_train, y_train, w_train, X_test, y_test, w_test, 
                        X_eval, customer_id_eval, working_dir, experiment_name, feature_names=None, X_train_full=None, y_train_full=None, w_train_full=None
                        ):
    """
    Función principal que replica el experimento con múltiples semillas.

    Parameters:
    -----------
    X_train, X_test, X_eval : array-like
        Features de entrenamiento, test y evaluación
    y_train, y_test : array-like
        Targets de entrenamiento y test
    w_train, w_test : array-like
        Pesos de entrenamiento y test
    customer_id_eval : array-like
        Números de cliente para el conjunto de evaluación
    X_train_full, y_train_full, w_train_full : array-like, optional
        Datos de entrenamiento sin sampling para el modelo final
        Si no se proporcionan, se usa X_train, y_train, w_train
    working_dir : str, optional
        Directorio de trabajo. Si None, usa el directorio del experimento llamador
    experiment_name : str

    Returns:
    --------
    dict
        Resultados del experimento con todas las semillas
    """

    print(f"🔬 Iniciando replicación para experimento: {experiment_name}")
    gc.collect()
    # Determinar qué datos usar para el modelo final
    if X_train_full is not None and y_train_full is not None and w_train_full is not None:
        # Usar datos sin sampling para el modelo final
        X_train_for_final = X_train_full
        y_train_for_final = y_train_full
        w_train_for_final = w_train_full
        print(f"🎯 Modelo final usará datos SIN sampling: {len(X_train_full)} registros de entrenamiento")
    else:
        # Usar datos con sampling para el modelo final (comportamiento anterior)
        X_train_for_final = X_train
        y_train_for_final = y_train
        w_train_for_final = w_train
        print(f"⚠️ Modelo final usará datos CON sampling: {len(X_train)} registros de entrenamiento")

    params = load_hyperparameters(working_dir, experiment_name)

    results = {
        'experiment_name': experiment_name,
        'seeds': RANDOM_SEEDS,
        'hyperparameters': params,
        'results_per_seed': [],
        'average_metrics': {}
    }

    eval_predictions = []
    aucs = []
    ganancias = []
    final_model = None  # Para guardar el último modelo final (para feature importance)
    final_X_shape = None  # Para conocer el número de features

    print(f"🎲 Entrenando con {len(RANDOM_SEEDS)} semillas diferentes...")

    for i, seed in enumerate(RANDOM_SEEDS):
        print(f"\n--- Semilla {i+1}/{len(RANDOM_SEEDS)}: {seed} ---")

        modelo, y_pred_test, auc, ganancia = train_model_single_seed(
            X_train, y_train, w_train, X_test, y_test, w_test, params, seed
        )

        print(f"  AUC: {auc:.6f}")
        print(f"  Ganancia: {ganancia:,.0f}")

        # Optimizar threshold usando predicciones en test set
        print("  🎯 Optimizando threshold...")
        threshold_result = optimize_threshold(y_pred_test, y_test, w_test)
        optimal_threshold = threshold_result['optimal_threshold']

        seed_result = {
            'seed': seed,
            'auc': auc,
            'ganancia': ganancia,
            'best_iteration': modelo.best_iteration,
            'optimal_threshold': optimal_threshold,
            'threshold_gain': threshold_result['best_gain']
        }
        results['results_per_seed'].append(seed_result)
        aucs.append(auc)
        ganancias.append(ganancia)

        print("  Entrenando modelo final con train+test...")

        X_final = np.vstack([X_train_for_final, X_test])
        y_final = np.concatenate([y_train_for_final, y_test])
        w_final = np.concatenate([w_train_for_final, w_test])

        # Obtener parámetros optimizados para modelo final
        base_params_final = params.copy()
        base_params_final.update({
            'seed': seed,
            'verbose': 0
        })
        
        params_final = get_optimized_lgb_params(base_params_final, quiet=True)
        
        # Limpiar cualquier parámetro relacionado con early stopping
        params_final.pop('early_stopping_rounds', None)
        
        # Obtener num_boost_round de los parámetros, o usar default
        num_boost_round = params_final.pop('num_boost_round', 10000)

        train_final_dataset = lgb.Dataset(X_final, label=y_final, weight=w_final)
        modelo_final = lgb.train(
            params_final,
            train_final_dataset,
            num_boost_round=num_boost_round
        )
        
        # Guardar el último modelo y info para feature importance
        final_model = modelo_final
        final_X_shape = X_final.shape

        y_pred_eval = modelo_final.predict(X_eval)
        eval_predictions.append(y_pred_eval)

        df_pred = pd.DataFrame({
            'numero_de_cliente': customer_id_eval,
            'probabilidad': y_pred_eval
        })

        pred_filename = f"{experiment_name}_{seed}.csv"
        pred_path = os.path.join(working_dir, pred_filename)
        df_pred.to_csv(pred_path, index=False)
        print(f"  Predicciones guardadas: {pred_filename}")
        gc.collect()

    # Calcular métricas de thresholds optimizados
    optimal_thresholds = [r['optimal_threshold'] for r in results['results_per_seed']]
    threshold_gains = [r['threshold_gain'] for r in results['results_per_seed']]
    
    results['average_metrics'] = {
        'auc_mean': np.mean(aucs),
        'auc_std': np.std(aucs),
        'gain_mean': np.mean(ganancias),
        'gain_std': np.std(ganancias),
        'threshold_mean': np.mean(optimal_thresholds),
        'threshold_std': np.std(optimal_thresholds),
        'threshold_gain_mean': np.mean(threshold_gains),
        'threshold_gain_std': np.std(threshold_gains)
    }
    
    # Threshold del ensemble (promedio de thresholds individuales)
    ensemble_threshold = np.mean(optimal_thresholds)
    results['ensemble_threshold'] = ensemble_threshold

    print(f"\n📊 RESULTADOS FINALES:")
    print(f"  AUC promedio: {results['average_metrics']['auc_mean']:.6f} ± {results['average_metrics']['auc_std']:.6f}")
    print(f"  Ganancia promedio: {results['average_metrics']['gain_mean']:,.0f} ± {results['average_metrics']['gain_std']:,.0f}")
    print(f"  🎯 Threshold promedio: {results['average_metrics']['threshold_mean']:.4f} ± {results['average_metrics']['threshold_std']:.4f}")
    print(f"  💰 Ganancia threshold promedio: {results['average_metrics']['threshold_gain_mean']:,.0f} ± {results['average_metrics']['threshold_gain_std']:,.0f}")

    results_filename = f"results_{experiment_name}.json"
    results_path = os.path.join(working_dir, results_filename)    

    results = convertir_a_nativo(results)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"📄 Resultados guardados: {results_filename}")

    # Generar gráfico de feature importance del modelo final
    if final_model is not None and final_X_shape is not None:
        print("\n📊 Generando gráfico de feature importance...")
        try:
            if feature_names is not None and len(feature_names) == final_X_shape[1]:
                plot_feature_names = feature_names
            else:
                plot_feature_names = [f'feature_{i}' for i in range(final_X_shape[1])]
            
            plot_feature_importance(
                model=final_model,
                feature_names=plot_feature_names,
                top_features=30,
                experiment_name=experiment_name,
                working_dir=working_dir,
            )
        except Exception as e:
            print(f"⚠️ Error generando feature importance: {e}")
        gc.collect()
    print("\n🎯 Creando ensemble...")
    ensemble_predictions = np.mean(eval_predictions, axis=0)

    df_ensemble = pd.DataFrame({
        'numero_de_cliente': customer_id_eval,
        'probabilidad': ensemble_predictions
    })

    ensemble_filename = f"{experiment_name}_ensemble.csv"
    ensemble_path = os.path.join(working_dir, ensemble_filename)
    df_ensemble.to_csv(ensemble_path, index=False)

    print(f"🏆 Ensemble guardado: {ensemble_filename}")

    return results