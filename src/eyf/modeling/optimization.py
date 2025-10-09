"""
Funciones para optimización de hiperparámetros con Optuna
"""
import os
import time

import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np

from .callbacks import lgb_auc_eval
from ..utils.data_dict import GANANCIA_ACIERTO, COSTO_ESTIMULO


def detect_hardware_capabilities():
    """
    Detecta las capacidades de hardware disponibles.
    
    Returns:
    --------
    dict
        Diccionario con información de hardware
    """
    capabilities = {
        'num_cores': os.cpu_count(),
        'has_gpu': False,
        'device_type': 'cpu'
    }
    
    # Intentar detectar GPU (requiere OpenCL)
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        if platforms:
            capabilities['has_gpu'] = True
            capabilities['device_type'] = 'gpu'
            print(f"🚀 GPU detectada: {platforms[0].name}")
    except ImportError:
        print("💡 GPU optimization disponible instalando: pip install pyopencl")
    except Exception:
        pass  # GPU no disponible o no compatible
    
    print(f"🔧 Hardware detectado: {capabilities['num_cores']} cores CPU")
    return capabilities


def create_optuna_objective(hyperparameter_space, X_train, y_train, w_train, seed=None, n_folds=5, feval=None):
    """
    Crea la función objetivo para Optuna.
    
    Parameters:
    -----------
    hyperparameter_space : dict
        Espacio de hiperparámetros a optimizar
    X_train : array-like
        Features de entrenamiento
    y_train : array-like
        Target de entrenamiento
    w_train : array-like
        Pesos de entrenamiento
    seed : int, optional
        Semilla para reproducibilidad
    n_folds : int, default=5
        Número de folds para cross-validation
    feval : callable, optional
        Función de evaluación personalizada. Si None, usa lgb_auc_eval
        
    Returns:
    --------
    callable
        Función objetivo para Optuna
    """
    # Usar lgb_auc_eval por defecto
    if feval is None:
        feval = lgb_auc_eval
    
    # Determinar nombre de métrica desde feval
    if hasattr(feval, '__name__'):
        metric_name = feval.__name__.replace('lgb_', '').replace('_eval', '')
    else:
        metric_name = 'metric'
    
    # Detectar capacidades de hardware una sola vez
    hw_capabilities = detect_hardware_capabilities()
    
    def objective(trial):
        trial_start_time = time.time()
        
        # Configurar parámetros base con optimizaciones de hardware
        params = {
            'objective': 'binary',
            'metric': 'None',  # Usamos métrica personalizada
            'boosting_type': 'gbdt',
            'verbose': -1,
            
            # Optimizaciones de hardware automáticas
            'device_type': hw_capabilities['device_type'],  # CPU o GPU
            'num_threads': hw_capabilities['num_cores'] // 2,  # Usar la mitad de cores para evitar saturación durante CV
            'force_row_wise': True,              # Optimización de memoria
            'max_bin': 255,                      # Máximo número de bins para mejor precisión
            'bin_construct_sample_cnt': 200000,  # Muestras para construir histogramas
            'data_sample_strategy': 'bagging',   # Estrategia optimizada de sampling
            'feature_pre_filter': False,         # No filtrar features automáticamente
            'max_cat_threshold': 32,             # Threshold para categorías automáticas
            'cat_smooth': 10,                    # Smoothing para features categóricas
            
            'seed': seed or 42
        }
        
        # Parámetros adicionales para GPU si está disponible
        if hw_capabilities['has_gpu']:
            params.update({
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'min_data_in_leaf': max(20, params.get('min_data_in_leaf', 20))  # GPU requiere más datos por hoja
            })
        #print(hyperparameter_space)
        # Samplear hiperparámetros
        for param_name, (suggest_type, min_val, max_val) in hyperparameter_space.items():
            if suggest_type == 'int':
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
            elif suggest_type == 'float':
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            elif suggest_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, min_val)  # min_val es la lista de opciones
        
        # Crear dataset
        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
        
        try:
            # Extraer parámetros especiales que no van directamente a LightGBM
            early_stopping_rounds = params.pop('early_stopping_rounds', 50)
            num_boost_round = params.pop('num_boost_round', 1000)
            
            # Realizar cross-validation con timeout
            cv_results = lgb.cv(
                params,
                train_data,
                num_boost_round=num_boost_round,
                folds=None,
                nfold=n_folds,
                stratified=True,
                shuffle=True,
                feval=feval,
                seed=seed,
                return_cvbooster=False,
                callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
            )
            
            # Obtener el mejor score
            metric_scores = cv_results[f'valid {metric_name}-mean']
            best_score = max(metric_scores)
            
            return best_score
            
        except Exception as e:
            print(f"❌ Error en trial {trial.number}: {e}")
    
    return objective


def optimize_hyperparameters_with_optuna(hyperparameter_space, X_train, y_train, w_train, n_trials=100, seed=None, study_name=None, feval=None):
    """
    Optimiza hiperparámetros usando Optuna.
    
    Parameters:
    -----------
    hyperparameter_space : dict
        Espacio de hiperparámetros a optimizar
    X_train : array-like
        Features de entrenamiento
    y_train : array-like
        Target de entrenamiento
    w_train : array-like
        Pesos de entrenamiento
    n_trials : int, default=100
        Número de trials de optimización
    seed : int, optional
        Semilla para reproducibilidad
    study_name : str, optional
        Nombre del estudio para logging
    feval : callable, optional
        Función de evaluación personalizada. Si None, usa lgb_auc_eval
        
    Returns:
    --------
    dict
        Mejores hiperparámetros encontrados
    """
    print(f"🚀 Iniciando optimización bayesiana con {n_trials} trials...")
    
    # Usar lgb_auc_eval por defecto
    if feval is None:
        feval = lgb_auc_eval
    
    # Determinar nombre de métrica desde feval
    if hasattr(feval, '__name__'):
        metric_name = feval.__name__.replace('lgb_', '').replace('_eval', '')
    else:
        metric_name = 'metric'
    
    # Crear estudio con pruning
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Crear función objetivo
    objective = create_optuna_objective(
        hyperparameter_space, X_train, y_train, w_train, seed, feval=feval
    )
    
    # Optimizar con paralelización conservadora si hay suficientes cores
    start_time = time.time()
    n_jobs = 1  # Por defecto secuencial
    
    # Usar paralelización solo si hay más de 4 cores y más de 10 trials
    if os.cpu_count() > 4 and n_trials > 10:
        n_jobs = min(2, os.cpu_count() // 4)  # Máximo 2 jobs paralelos, conservador
        print(f"🔄 Usando {n_jobs} procesos paralelos para optimización")
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    total_time = time.time() - start_time
    
    # Mostrar resultados
    print(f"\n✅ Optimización completada en {total_time/60:.1f} minutos")
    print(f"📊 Mejor {metric_name}: {study.best_value:.6f}")
    print("🎯 Mejores hiperparámetros:")
    
    for param, value in study.best_params.items():
        print(f"   {param}: {value}")
    
    return study

def optimize_threshold(y_prob, y_true, weights=None, threshold_range=(0.01, 0.1), n_points=100):
    """
    Optimiza el threshold para maximizar la ganancia.
    
    Parameters:
    -----------
    y_prob : array-like
        Probabilidades predichas
    y_true : array-like
        Etiquetas verdaderas (0/1)
    weights : array-like, optional
        Pesos de las muestras
    threshold_range : tuple, optional
        Rango de thresholds a probar (min, max)
    n_points : int, optional
        Número de puntos a evaluar en el rango
        
    Returns:
    --------
    dict
        Diccionario con threshold óptimo y métricas
    """
    if weights is None:
        weights = np.ones(len(y_true))
    
    # Generar thresholds a probar
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_points)
    
    best_threshold = None
    best_gain = float('-inf')
    results = []
    
    print(f"🔍 Probando {n_points} thresholds en rango [{threshold_range[0]:.3f}, {threshold_range[1]:.3f}]")
    
    for threshold in thresholds:
        # Aplicar threshold
        y_pred_binary = (y_prob >= threshold).astype(int)
        #print((y_true == 1) & (y_pred_binary == 1))
        
        # Calcular métricas ponderadas
        tp = np.sum(((y_true == 1) & (y_pred_binary == 1)).astype(int) * weights)
        fp = np.sum(((y_true == 0) & (y_pred_binary == 1)).astype(int) * weights)
        tn = np.sum(((y_true == 0) & (y_pred_binary == 0)).astype(int) * weights)
        fn = np.sum(((y_true == 1) & (y_pred_binary == 0)).astype(int) * weights)
        
        # Calcular ganancia
        ganancia = tp * GANANCIA_ACIERTO - fp * COSTO_ESTIMULO
        
        # Calcular otras métricas
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'ganancia': ganancia,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        if ganancia > best_gain:
            best_gain = ganancia
            best_threshold = threshold
    
    best_result = next(r for r in results if r['threshold'] == best_threshold)
    
    print(f"✅ Threshold óptimo: {best_threshold:.4f}")
    print(f"💰 Ganancia máxima: {best_gain:,.0f}")
    print(f"📊 Precision: {best_result['precision']:.4f}, Recall: {best_result['recall']:.4f}")
    
    return {
        'optimal_threshold': best_threshold,
        'best_gain': best_gain,
        'best_metrics': best_result,
        'all_results': results
    }

def save_trials(study, study_name, working_dir):
    trials_data = []
    for trial in study.trials:
        trial_dict = {
            'trial_number': trial.number,
            'value': trial.value,
            'state': trial.state.name,
            'best_iter': trial.user_attrs.get('best_iter', None),
            'duration_minutes': trial.user_attrs.get('duration_minutes', None)
        }
        # Agregar hiperparámetros
        trial_dict.update(trial.params)
        trials_data.append(trial_dict)
    
    trials_df = pd.DataFrame(trials_data)
    trials_csv_path = os.path.join(working_dir, f"{study_name}_trials.csv")
    trials_df.to_csv(trials_csv_path, index=False)
    print(f"💾 Trials guardados en: {trials_csv_path}")