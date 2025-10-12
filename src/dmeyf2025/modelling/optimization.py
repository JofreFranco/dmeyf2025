import time
import lightgbm as lgb
import logging

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

    logger.info(f"Training dataset shape: {X_train.columns}")
    # Determinar nombre de métrica desde feval
    if hasattr(feval, '__name__'):
        metric_name = feval.__name__.replace('lgb_', '').replace('_eval', '')
    else:
        metric_name = 'metric'
    
    X_train = X_train.copy()
    def objective(trial):
        trial_start_time = time.time()
        
        # Configurar parámetros base con optimizaciones de hardware
        

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


            metric_scores = cv_results[f'valid {metric_name}-mean']
            best_score = max(metric_scores)
            
            best_iteration = len(metric_scores)
            trial.set_user_attr('metric_scores', metric_scores)
            trial.set_user_attr('actual_num_boost_round', best_iteration)
            
            trial.set_user_attr('AUC', cv_results['valid auc-mean'][0])
            trial.set_user_attr('AUC-std', cv_results['valid auc-stdv'][0])
            trial.set_user_attr('Gain', cv_results['valid gan-mean'][0])
            trial.set_user_attr('Gain-std', cv_results['valid gan-stdv'][0])
            trial.set_user_attr('Logloss', cv_results['valid binary_logloss-mean'][0])
            trial.set_user_attr('Logloss-std', cv_results['valid binary_logloss-stdv'][0])
            return best_score
            
        except Exception as e:
            logger.error(f"❌ Error en trial {trial.number}: {e}")
    
    return objective
