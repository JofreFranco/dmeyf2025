import time
import lightgbm as lgb
import logging
import json
import optuna

from dmeyf2025.utils.save_study import save_trials
from dmeyf2025.metrics.revenue import lgb_gan_eval

logger = logging.getLogger(__name__)

def create_optuna_objective(hyperparameter_space, X_train, y_train, w_train = None, seed=None, n_folds=5, feval=None, params=None):
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
        Función de evaluación personalizada. Si None, usa AUC
        
    Returns:
    --------
    callable
        Función objetivo para Optuna
    """
    # Determinar nombre de métrica desde feval
    if hasattr(feval, '__name__'):
        metric_name = feval.__name__.replace('lgb_', '').replace('_eval', '')
    else:
        metric_name = 'metric'
    
    X_train = X_train.copy()
    def objective(trial):
        trial_start_time = time.time()
        
        trial_params = params.copy()

        for param_name, (suggest_type, min_val, max_val) in hyperparameter_space.items():
            if param_name == 'learning_rate':
                trial_params[param_name] = trial.suggest_float(param_name, min_val, max_val, log=True)
            else:
                if suggest_type == 'int':
                    trial_params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                elif suggest_type == 'float':
                    trial_params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                elif suggest_type == 'categorical':
                    trial_params[param_name] = trial.suggest_categorical(param_name, min_val)  # min_val es la lista de opciones
        

        # Crear dataset

        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
        
        
        try:
            # Extraer parámetros especiales que no van directamente a LightGBM
            early_stopping_rounds = trial_params.pop('early_stopping_rounds', 50)
            num_boost_round = trial_params.pop('num_boost_round', 1000)
            
            # Realizar cross-validation con timeout
            cv_results = lgb.cv(
                trial_params,
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

            # TODO: Revisar si esto es correcto
            metric_scores = cv_results[f'valid {metric_name}-mean']
            best_score = max(metric_scores)
            
            best_iteration = len(metric_scores)
            trial.set_user_attr('metric_scores', metric_scores)
            trial.set_user_attr('actual_num_boost_round', best_iteration)
            
            trial.set_user_attr('AUC', cv_results['valid auc-mean'][-1])
            trial.set_user_attr('AUC-std', cv_results['valid auc-stdv'][-1])
            trial.set_user_attr('Gain', cv_results['valid gan-mean'][-1])
            trial.set_user_attr('Gain-std', cv_results['valid gan-stdv'][-1])
            trial.set_user_attr('Logloss', cv_results['valid binary_logloss-mean'][-1])
            trial.set_user_attr('Logloss-std', cv_results['valid binary_logloss-stdv'][-1])
            return best_score
            
        except Exception as e:
            logger.error(f"❌ Error en trial {trial.number}: {e}")
    
    return objective

def optimize_params(experiment_config, X_train, y_train, seed = 42):
    params = {
            'metric': ['auc', 'binary_logloss'],
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'device_type': "CPU",  # CPU o GPU
            'num_threads': 10,
            'force_row_wise': True,
            'max_bin': 31,
            'max_cat_threshold': 32,
            'cat_smooth': 10,
            'seed': seed
        }

    start_time = time.time()
    study = optuna.create_study(
        direction='maximize',
        study_name=experiment_config['experiment_name'],
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=experiment_config["n_init"])
    )
    # Crear función objetivo
    objective = create_optuna_objective(
        experiment_config["hyperparameter_space"], X_train, y_train, seed=seed, feval=lgb_gan_eval,
        params=params,
    )
    experiment_path = f"{experiment_config['experiments_path']}/{experiment_config['experiment_folder']}"
    study.optimize(objective, n_trials=experiment_config["n_trials"])
    
    total_time = time.time() - start_time

    logger.info(f"\n✅ Optimización completada en {total_time/60:.1f} minutos")
    logger.info(f"📊 Mejor ganancia: {study.best_value:.6f}")

    best_trial = study.best_trial
    auc = best_trial.intermediate_values.get(0)
    binary_loss = best_trial.intermediate_values.get(1)
    logger.info(f"📊 Métricas del mejor trial:")
    logger.info(f"Best AUC: {study.best_trial.user_attrs.get('AUC')}")
    logger.info(f"Best AUC-std: {study.best_trial.user_attrs.get('AUC-std')}")
    logger.info(f"Best Gain: {study.best_trial.user_attrs.get('Gain')}")
    logger.info(f"Best Gain-std: {study.best_trial.user_attrs.get('Gain-std')}")
    logger.info(f"Best Logloss: {study.best_trial.user_attrs.get('Logloss')}")
    logger.info(f"Best Logloss-std: {study.best_trial.user_attrs.get('Logloss-std')}")
    results = {
        "best_trial_number": study.best_trial.number,
        "best_trial_value": study.best_trial.value,
        "best_trial_AUC": study.best_trial.user_attrs.get('AUC'),
        "best_trial_AUC-std": study.best_trial.user_attrs.get('AUC-std'),
        "best_trial_Gain": study.best_trial.user_attrs.get('Gain'),
        "best_trial_Gain-std": study.best_trial.user_attrs.get('Gain-std'),
        "best_trial_Logloss": study.best_trial.user_attrs.get('Logloss'),
        "best_trial_Logloss-std": study.best_trial.user_attrs.get('Logloss-std'),
    }
    json_filename = f"results.json"
    json_path = f"{experiment_path}/{json_filename}"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Hiperparámetros guardados en: {json_filename}")
    best_params = study.best_params
    best_params.update(params)
    best_params.pop('early_stopping_rounds')
    
    # Obtener el número real de iteraciones del mejor trial
    best_trial = study.best_trial
    actual_num_boost_round = best_trial.user_attrs.get('actual_num_boost_round', best_params.get('num_boost_round', 1000))
    best_params['num_boost_round'] = actual_num_boost_round
    
    logger.info(f"🎯 Número real de iteraciones usadas: {actual_num_boost_round}")

    json_filename = f"best_params.json"
    json_path = f"{experiment_path}/{json_filename}"
    
    with open(json_path, 'w') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Hiperparámetros guardados en: {json_filename}")
    save_trials(study, experiment_path)
    return best_params, study