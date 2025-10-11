"""
Experimento delta-lags2: Evaluación de features con delta 2 y lag 2
"""
from datetime import datetime
import json
import gc
import logging
from nturl2path import pathname2url
import time
import os
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
import pandas as pd

from dmeyf2025.experiments import experiment_init
from dmeyf2025.etl import ETL
from dmeyf2025.processors.target_processor import BinaryTargetProcessor, CreateTargetProcessor
from dmeyf2025.processors.sampler import SamplerProcessor
from dmeyf2025.processors.feature_processors import DeltaLagTransformer
from dmeyf2025.modelling.optimization import create_optuna_objective
from dmeyf2025.metrics.revenue import lgb_gan_eval, revenue_from_prob
from dmeyf2025.utils.save_study import save_trials
from dmeyf2025.modelling.train_model import train_models
from dmeyf2025.processors.eval_processors import scale

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


########## MAIN ##########
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experiment with specified config file.")
    parser.add_argument('--config', type=str, default='config1.yaml', help='YAML config file to load')
    args = parser.parse_args()
    config_file = args.config
    experiment_config = experiment_init(config_file, script_file=__file__, debug=False)
    DEBUG = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    seeds = experiment_config["seeds"]
    logger.info(f"""\n{'=' * 70}
    📅 {date_time}
    📝 Iniciando experimento: {experiment_config['experiment_name']}
    🎯 Descripción: {experiment_config['config']['experiment']['description']}
    🔧 Experiment folder: {experiment_config['experiment_folder']}
{'=' * 70}""")

    start_time = time.time()
   
    #### ETL ####
    """
    Lee los datos, calcula target ternario, y divide en train, test y eval.
    """
    etl = ETL(experiment_config['raw_data_path'], CreateTargetProcessor(), train_months = [202101, 202102, 202103, 202104, 202105, 202106],)
    X, y, _,_,_,_ = etl.execute_complete_pipeline()
    
    
    #### TRAIN PROCESSING ####
    target_processor = BinaryTargetProcessor(experiment_config['config']['experiment']['positive_classes'])
    X, y = target_processor.fit_transform(X, y)
    X["label"] = y

    #### Features ####
    delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2)
    X_transformed = delta_lag_transformer.fit_transform(X)
    logger.info(f"X_transformed.shape: {X_transformed.shape}")
    
    # Loggea todas las columnas, una por línea
    if False:
        logger.info("X_train_transformed.columns ({} columnas):\n{}".format(
            len(X_train_transformed_sampled.columns),
            "\n".join(X_train_transformed_sampled.columns)
        ))
    X_train = X_transformed[X_transformed["foto_mes"].isin(experiment_config['train_months'])]
    y_train = X_train["label"]
    X_train = X_train.drop(columns=["label"])
    X_eval = X_transformed[X_transformed["foto_mes"].isin([experiment_config['eval_month']])]
    X_eval = X_eval.drop(columns=["label"])
    X_test = X_transformed[X_transformed["foto_mes"].isin([experiment_config['test_month']])]
    #### OPTIMIZACIÓN DE HIPERPARÁMETROS ####
    sampler_processor = SamplerProcessor(experiment_config['SAMPLE_RATIO'])
    X_train_sampled, y_train_sampled = sampler_processor.fit_transform(X_train, y_train)
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
            'seed': seeds[0]
        }

    logger.info("Iniciando optimización...")

    start_time = time.time()
    study = optuna.create_study(
        direction='maximize',
        study_name=experiment_config['experiment_name'],
        sampler=optuna.samplers.TPESampler(seed=seeds[0], n_startup_trials=experiment_config["n_init"])
    )
    # Crear función objetivo
    objective = create_optuna_objective(
        experiment_config["hyperparameter_space"], X_train_sampled, y_train_sampled, seed=seeds[0], feval=lgb_gan_eval,
        params=params,
    )

    study.optimize(objective, n_trials=experiment_config["n_trials"], n_jobs=-1)

    # Mostrar resultados
    total_time = time.time() - start_time

    logger.info(f"\n✅ Optimización completada en {total_time/60:.1f} minutos")
    logger.info(f"📊 Mejor ganancia: {study.best_value:.6f}")
    logger.info("🎯 Mejores hiperparámetros:")
    
    for param, value in study.best_params.items():
        logger.info(f"   {param}: {value}")

    # Guardar best params y trials
    best_params = study.best_params
    best_params.update(params)
    best_params.pop('early_stopping_rounds')
    
    # Obtener el número real de iteraciones del mejor trial
    best_trial = study.best_trial
    actual_num_boost_round = best_trial.user_attrs.get('actual_num_boost_round', best_params.get('num_boost_round', 1000))
    best_params['num_boost_round'] = actual_num_boost_round
    
    logger.info(f"🎯 Número real de iteraciones usadas: {actual_num_boost_round}")

    json_filename = f"best_params.json"
    experiment_path = f"{experiment_config['experiments_path']}/{experiment_config['experiment_folder']}"
    json_path = f"{experiment_path}/{json_filename}"
    
    with open(json_path, 'w') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Hiperparámetros guardados en: {json_filename}")
    save_trials(study, experiment_path)
    gc.collect()

    #### Final Modeling #######################################
    ###########################################################
    logger.info("Iniciando Modelado final...")

    if not DEBUG:
        X_final_train = X_train
        y_final_train = y_train
    else:
        X_final_train = X_train_sampled
        y_final_train = y_train_sampled
    X_final_train.set_index("numero_de_cliente", inplace=True)
    X_eval.set_index("numero_de_cliente", inplace=True)
    logger.info(f"X_train.shape final training: {X_final_train.shape}")
    n_seeds = len(seeds)
    logger.info(f"Entrenando y prediciendo con {n_seeds} seeds para ensamblado...")
    
    predictions,models = train_models(X_final_train, y_final_train, X_eval, best_params, seeds, experiment_path)

    #OPTIMIZACIÓN DE ENVÍOS
    if False:
        logger.info("Iniciando optimización de envíos...")
        min_sends = 1000
        max_sends = 30000
        best_n_sends = []
        for model in models:
            y_pred = model.predict(X_train)
            best_sends = sends_optimization(y_pred, y_train, min_sends, max_sends)
            best_n_sends.append(best_sends)

        logger.info(f"🎯 Mejor número de envíos: {np.mean(best_n_sends)} ± {np.std(best_n_sends)}")
        n_sends = int(np.mean(best_n_sends))
    else:
        n_sends = 11500
    ####
    
    ones = np.ones(n_sends, dtype=int)
    zeros = np.zeros(len(predictions)-n_sends, dtype=int)
    sends = np.concatenate([ones, zeros])
    predictions["predicted"] = sends
    predictions[["numero_de_cliente", "predicted"]].to_csv(f"{experiment_path}/{experiment_config["experiment_folder"]}_ensemble_predictions.csv", index=False)

    
    
    # Scale median
    X_scaled = scale(X, strategy="median")
    delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2)
    X_transformed = delta_lag_transformer.fit_transform(X_scaled)
    X_transformed.set_index("numero_de_cliente", inplace=True)
    X_transformed.loc[:, "label"] = y
    X_eval = X_transformed[X_transformed["foto_mes"].isin([experiment_config['eval_month']])]
    X_eval = X_eval.drop(columns=["label"])
    print(set(X_final_train.columns) - set(X_eval.columns))
    logger.info(f"X_eval.shape: {X_eval.shape}")
    logger.info(f"X_train.shape: {X_final_train.shape}")

    predictions = pd.DataFrame()
    for n, model in enumerate(models):
        predictions["numero_de_cliente"] = X_eval.index
        y_pred = model.predict(X_eval)
        predictions[f"pred_{n}"] = y_pred
    ones = np.ones(n_sends, dtype=int)
    zeros = np.zeros(len(predictions)-n_sends, dtype=int)
    sends = np.concatenate([ones, zeros])
    predictions["predicted"] = sends
    predictions[["numero_de_cliente", "predicted"]].to_csv(f"{experiment_path}/{experiment_config["experiment_folder"]}_ensemble_predictions_scaled_median.csv", index=False)
    # Scale mean
    X_scaled = scale(X, strategy="mean")
    delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2)
    X_transformed = delta_lag_transformer.fit_transform(X_scaled)
    X_transformed.set_index("numero_de_cliente", inplace=True)
    X_transformed.loc[:, "label"] = y
    X_eval = X_transformed[X_transformed["foto_mes"].isin([experiment_config['eval_month']])]
    X_eval = X_eval.drop(columns=["label"])

    predictions = pd.DataFrame()
    for n, model in enumerate(models):
        predictions["numero_de_cliente"] = X_eval.index
        y_pred = model.predict(X_eval)
        predictions[f"pred_{n}"] = y_pred
    ones = np.ones(n_sends, dtype=int)
    zeros = np.zeros(len(predictions)-n_sends, dtype=int)
    sends = np.concatenate([ones, zeros])
    predictions["predicted"] = sends
    predictions[["numero_de_cliente", "predicted"]].to_csv(f"{experiment_path}/{experiment_config["experiment_folder"]}_ensemble_predictions_scaled_mean.csv", index=False)

    #### Final Modeling Con Escalado de Hiperparámetros ####
    ########################################################
    logger.info("Iniciando Modelado final con escalado de hiperparámetros...")
    best_params["min_data_in_leaf"] = int(best_params["min_data_in_leaf"]/experiment_config['SAMPLE_RATIO'])
    
    predictions,models = train_models(X_final_train, y_final_train, X_eval, best_params, seeds, experiment_path)

    ones = np.ones(n_sends, dtype=int)
    zeros = np.zeros(len(predictions)-n_sends, dtype=int)
    sends = np.concatenate([ones, zeros])
    predictions["predicted"] = sends
    predictions[["numero_de_cliente", "predicted"]].to_csv(f"{experiment_path}/{experiment_config["experiment_folder"]}_ensemble_predictions_hpscaled.csv", index=False)


    # Scale median
    X_scaled = scale(X, strategy="median")
    delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2)
    X_transformed = delta_lag_transformer.fit_transform(X_scaled)
    X_transformed.set_index("numero_de_cliente", inplace=True)
    X_transformed.loc[:, "label"] = y
    X_eval = X_transformed[X_transformed["foto_mes"].isin([experiment_config['eval_month']])]
    X_eval = X_eval.drop(columns=["label"])

    predictions = pd.DataFrame()
    for n, model in enumerate(models):
        predictions["numero_de_cliente"] = X_eval.index
        y_pred = model.predict(X_eval)
        predictions[f"pred_{n}"] = y_pred
    ones = np.ones(n_sends, dtype=int)
    zeros = np.zeros(len(predictions)-n_sends, dtype=int)
    sends = np.concatenate([ones, zeros])
    predictions["predicted"] = sends
    predictions[["numero_de_cliente", "predicted"]].to_csv(f"{experiment_path}/{experiment_config["experiment_folder"]}_ensemble_predictions_scaled_median_hpscaled.csv", index=False)

    # Scale mean
    X_scaled = scale(X, strategy="mean")
    delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2)
    X_transformed = delta_lag_transformer.fit_transform(X_scaled)
    X_transformed.set_index("numero_de_cliente", inplace=True)
    X_transformed.loc[:, "label"] = y
    X_eval = X_transformed[X_transformed["foto_mes"].isin([experiment_config['eval_month']])]
    X_eval = X_eval.drop(columns=["label"])

    predictions = pd.DataFrame()
    for n, model in enumerate(models):
        predictions["numero_de_cliente"] = X_eval.index
        y_pred = model.predict(X_eval)
        predictions[f"pred_{n}"] = y_pred
    ones = np.ones(n_sends, dtype=int)
    zeros = np.zeros(len(predictions)-n_sends, dtype=int)
    sends = np.concatenate([ones, zeros])
    predictions["predicted"] = sends
    predictions[["numero_de_cliente", "predicted"]].to_csv(f"{experiment_path}/{experiment_config["experiment_folder"]}_ensemble_predictions_scaled_mean_hpscaled.csv", index=False)
    logger.info(f"Experimento completado en {(time.time() - start_time)/60:.2f} minutos")

