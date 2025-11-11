import time
import lightgbm as lgb
import logging
import json
import optuna
import numpy as np

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
    if "weight" in X_train.columns or "label" in X_train.columns or "target" in X_train.columns:
        logger.warning("Weight, label or target column found in X_train, removing it")
        X_train = X_train.drop(columns=["weight", "label", "target"])
    
    n_train = len(X_train)
    
    def objective(trial):
        trial_start_time = time.time()
        
        trial_params = params.copy()

        for param_name, (suggest_type, min_val, max_val) in hyperparameter_space.items():
            if param_name == 'learning_rate':
                trial_params[param_name] = trial.suggest_float(param_name, min_val, max_val, log=True)
            elif param_name == 'rel_min_data_in_leaf':
                # Hiperparámetro relativo: min_data_in_leaf = rel_min_data_in_leaf * len(X_train)
                rel_value = trial.suggest_float(param_name, min_val, max_val, log=True)
                abs_value = int(max(1, int(rel_value * n_train)))
                trial_params['min_data_in_leaf'] = abs_value
                trial.set_user_attr('rel_min_data_in_leaf', rel_value)
                trial.set_user_attr('min_data_in_leaf_calculated', abs_value)
            elif param_name == 'rel_num_leaves':
                # Hiperparámetro relativo: num_leaves = 2 + rel_num_leaves * len(X_train) / min_data_in_leaf
                rel_value = trial.suggest_float(param_name, min_val, max_val, log=True)
                min_data_in_leaf = trial_params.get('min_data_in_leaf', 1)
                abs_value = int(2 + rel_value * n_train / min_data_in_leaf)
                trial_params['num_leaves'] = abs_value
                trial.set_user_attr('rel_num_leaves', rel_value)
                trial.set_user_attr('num_leaves_calculated', abs_value)
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
            early_stopping_rounds = trial_params.pop('early_stopping_rounds', 750)
            num_boost_round = trial_params.pop('num_boost_round', 15)
            num_boost_round = int(num_boost_round/trial_params['learning_rate'])
            print(early_stopping_rounds, num_boost_round)
            if num_boost_round > early_stopping_rounds:
                if num_boost_round/10 < 2000:
                    early_stopping_rounds = int(num_boost_round/10)
                else:
                    early_stopping_rounds = 2000
            # Realizar cross-validation con timeout
            cv_results = lgb.cv(
                trial_params,
                train_data,
                num_boost_round=num_boost_round,
                folds=None,
                nfold=n_folds,
                stratified=True,
                shuffle=True,
                feval=lgb_gan_eval,
                seed=seed,
                return_cvbooster=False,
                callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
            )

            metric_scores = cv_results[f'valid {metric_name}-mean']
            best_score = np.median(metric_scores)
            best_iteration = len(metric_scores)
            trial.set_user_attr('metric_scores', metric_scores)
            trial.set_user_attr('actual_num_boost_round', best_iteration)
            #trial.set_user_attr('best_k', int(np.mean(feval_obj.best_ks)))
            trial.set_user_attr('AUC', cv_results['valid auc-mean'][-1])
            trial.set_user_attr('AUC-std', cv_results['valid auc-stdv'][-1])
            trial.set_user_attr('Gain', cv_results['valid gan-mean'][-1])
            trial.set_user_attr('Gain-std', cv_results['valid gan-stdv'][-1])
            #trial.set_user_attr('Logloss', cv_results['valid binary_logloss-mean'][-1])
            #trial.set_user_attr('Logloss-std', cv_results['valid binary_logloss-stdv'][-1])
            return best_score
            
        except Exception as e:
            logger.error(f"❌ Error en trial {trial.number}: {e}")
    
    return objective

def optimize_params(experiment_config, X_train, y_train, w_train, seed = 42):
    params = {
            'metric': 'auc',
            'objective': 'binary',
            'first_metric_only': True,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'max_depth': -1,
            'min_gain_to_split': 0,
            'min_sum_hessian_in_leaf': 0.001,
            'lambda_l1': 0,
            'lambda_l2': 0,
            'scale_pos_weight': 1,
            'is_unbalance': False,
            'boosting_type': 'gbdt',
            'verbose': -100,
            'extra_trees': False,
            'device_type': "CPU",  # CPU o GPU
            'num_threads': 10,
            'force_row_wise': True,
            'max_bin': 31,
            'seed': seed
        }
    start_time = time.time()
    study = optuna.create_study(
        direction='maximize',
        study_name=experiment_config['experiment_name'],
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=experiment_config["n_init"])
    )
    # Crear función objetivo
    if len(w_train) != len(X_train):
        raise ValueError("w_train and X_train must have the same length")
    if len(w_train) != len(y_train):
        raise ValueError("w_train and y_train must have the same length")
    if len(w_train) != len(X_train):
        raise ValueError("w_train and X_train must have the same length")
    if len(w_train) != len(y_train):
        raise ValueError("w_train and y_train must have the same length")
    if len(w_train) != len(X_train):
        raise ValueError("w_train and X_train must have the same length")
    objective = create_optuna_objective(
        experiment_config["hyperparameter_space"], X_train, y_train, w_train, seed=seed, feval=lgb_gan_eval,
        params=params,
    )
    experiment_path = experiment_config['experiment_dir']
    study.optimize(objective, n_trials=experiment_config["n_trials"])
    
    total_time = time.time() - start_time

    logger.info(f"\n✅ Optimización completada en {total_time/60:.1f} minutos")
    logger.info(f"📊 Mejor ganancia: {study.best_value:.6f}")

    best_trial = study.best_trial

    logger.info(f"📊 Métricas del mejor trial:")
    #logger.info(f"Best k: {study.best_trial.user_attrs.get('best_k')}")
    logger.info(f"Best AUC: {study.best_trial.user_attrs.get('AUC')}")
    logger.info(f"Best AUC-std: {study.best_trial.user_attrs.get('AUC-std')}")
    logger.info(f"Best Gain: {study.best_trial.user_attrs.get('Gain')}")
    logger.info(f"Best Gain-std: {study.best_trial.user_attrs.get('Gain-std')}")
    #logger.info(f"Best Logloss: {study.best_trial.user_attrs.get('Logloss')}")
    #logger.info(f"Best Logloss-std: {study.best_trial.user_attrs.get('Logloss-std')}")
    results = {
        "best_trial_number": study.best_trial.number,
        #"best_trial_best_k": study.best_trial.user_attrs.get('best_k'),
        "best_trial_value": study.best_trial.value,
        "best_trial_AUC": study.best_trial.user_attrs.get('AUC'),
        "best_trial_AUC-std": study.best_trial.user_attrs.get('AUC-std'),
        "best_trial_Gain": study.best_trial.user_attrs.get('Gain'),
        "best_trial_Gain-std": study.best_trial.user_attrs.get('Gain-std'),
        #"best_trial_Logloss": study.best_trial.user_attrs.get('Logloss'),
        #"best_trial_Logloss-std": study.best_trial.user_attrs.get('Logloss-std'),
    }
    json_filename = "results.json"
    json_path = experiment_path / json_filename
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Resultados guardados en: {json_filename}")
    best_params = study.best_params
    best_params.update(params)
    best_params.pop('early_stopping_rounds')
    
    # Obtener el número real de iteraciones del mejor trial
    best_trial = study.best_trial
    actual_num_boost_round = best_trial.user_attrs.get('actual_num_boost_round', best_params.get('num_boost_round', 1000))
    best_params['num_boost_round'] = actual_num_boost_round
    
    # Guardar valores relativos y absolutos calculados si existen
    if 'rel_min_data_in_leaf' in best_trial.user_attrs:
        best_params['rel_min_data_in_leaf'] = best_trial.user_attrs.get('rel_min_data_in_leaf')
        best_params['min_data_in_leaf'] = best_trial.user_attrs.get('min_data_in_leaf_calculated')
        logger.info(f"🎯 rel_min_data_in_leaf: {best_params['rel_min_data_in_leaf']:.6f} -> min_data_in_leaf: {best_params['min_data_in_leaf']}")
    
    if 'rel_num_leaves' in best_trial.user_attrs:
        best_params['rel_num_leaves'] = best_trial.user_attrs.get('rel_num_leaves')
        best_params['num_leaves'] = best_trial.user_attrs.get('num_leaves_calculated')
        logger.info(f"🎯 rel_num_leaves: {best_params['rel_num_leaves']:.6f} -> num_leaves: {best_params['num_leaves']}")
    
    #best_params['best_k'] = best_trial.user_attrs.get('best_k')
    logger.info(f"🎯 Número real de iteraciones usadas: {actual_num_boost_round}")

    json_filename = f"best_params.json"
    json_path = f"{experiment_path}/{json_filename}"
    
    with open(json_path, 'w') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Hiperparámetros guardados en: {json_filename}")
    save_trials(study, experiment_path)
    
    return best_params, study