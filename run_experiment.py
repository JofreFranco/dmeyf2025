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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


########## MAIN ##########
if __name__ == "__main__":
    experiment_config = experiment_init('config.yaml', script_file=__file__, debug=True)
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
    etl = ETL(experiment_config['raw_data_path'], CreateTargetProcessor(), experiment_config['config']['data']['train_months'], experiment_config['config']['data']['test_month'], experiment_config['config']['data']['eval_month'])
    X_train, y_train, X_test, y_test, X_eval, y_eval = etl.execute_complete_pipeline()
    
    #### TRAIN PROCESSING ####
    target_processor = BinaryTargetProcessor(experiment_config['config']['experiment']['positive_classes'])
    X_train, y_train = target_processor.fit_transform(X_train, y_train)
    sampler_processor = SamplerProcessor(experiment_config['SAMPLE_RATIO'])
    X_train_sampled, y_train_sampled = sampler_processor.fit_transform(X_train, y_train)

    #### Features ####
    delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2)
    X_train_transformed_sampled = delta_lag_transformer.fit_transform(X_train_sampled)
    X_test_transformed = delta_lag_transformer.transform(X_test)
    X_eval_transformed = delta_lag_transformer.transform(X_eval)

    logger.info(f"X_train_transformed.shape: {X_train_transformed_sampled.shape}")
    logger.info(f"X_test_transformed.shape: {X_test_transformed.shape}")
    logger.info(f"X_eval_transformed.shape: {X_eval_transformed.shape}")
    
    # Loggea todas las columnas, una por línea
    if False:
        logger.info("X_train_transformed.columns ({} columnas):\n{}".format(
            len(X_train_transformed_sampled.columns),
            "\n".join(X_train_transformed_sampled.columns)
        ))
    
    #### MODELING ####

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
        experiment_config["hyperparameter_space"], X_train_transformed_sampled, y_train_sampled, seed=seeds[0], feval=lgb_gan_eval,
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

    ### EVALUATION ###
    if X_test.shape[0] > 0:
        if DEBUG:
            X_test_transformed = X_train_transformed_sampled
            y_test = y_train_sampled


        auc_scores = []
        logloss_scores = []
        revenue_scores = []
        all_predictions = []
        all_true = []
        n_seeds = len(seeds)

        logger.info("Iniciando evaluación sobre todas las seeds...")
        start_time_eval = time.time()

        for i, seed in enumerate(seeds):
            logger.info(f"Seed {i+1}/{n_seeds} ({seed})")

            # Ajustamos la seed en los parámetros
            eval_params = dict(best_params)
            eval_params["seed"] = seed

            # Entrenar modelo
            train_final_dataset = lgb.Dataset(X_train_transformed_sampled, label=y_train_sampled)
            test_dataset = lgb.Dataset(X_test_transformed, label=y_test)
            model = lgb.train(eval_params, train_final_dataset, valid_sets=[test_dataset])
            y_pred = model.predict(X_test_transformed)

            auc = roc_auc_score(y_test, y_pred)
            revenue = revenue_from_prob(y_pred, y_test)
            try:
                logloss = log_loss(y_test, y_pred, labels=[0,1])
            except:
                logloss = None

            auc_scores.append(auc)
            revenue_scores.append(revenue)
            logloss_scores.append(logloss)
            all_predictions.append(y_pred)
            all_true.append(y_test)

            logger.info(f"Seed {seed} - AUC: {auc:.6f}, Revenue: {revenue:.2f}, LogLoss: {logloss if logloss is not None else 'N/A'}")

        total_eval_time = time.time() - start_time_eval

        # Agrupar predicciones y etiquetas verdaderas (por seed)
        all_predictions = np.stack(all_predictions)
        all_true = np.stack(all_true)

        logger.info(f"Evaluación completada en {total_eval_time:.2f} segundos sobre {n_seeds} seeds.")
        logger.info(f"AUC promedio: {np.mean(auc_scores):.6f} ± {np.std(auc_scores):.6f}")
        logger.info(f"Revenue promedio: {np.mean(revenue_scores):.2f} ± {np.std(revenue_scores):.2f}")

        # Si todos los logloss son calculables, reportar también
        logloss_valid = [x for x in logloss_scores if x is not None]
        if len(logloss_valid) > 0:
            logger.info(f"LogLoss promedio: {np.mean(logloss_valid):.6f} ± {np.std(logloss_valid):.6f}")
        else:
            logger.warning("No se pudo calcular LogLoss en ninguna seed.")
        
        # Ensamblar las predicciones promediando la probabilidad
        y_pred_ensemble = np.mean(all_predictions, axis=0)        
        logger.info(f"Evaluación del ensamblado de predicciones (promedio de probabilidad):")
        try:
            auc_ensemble = roc_auc_score(all_true[0], y_pred_ensemble)
        except:
            auc_ensemble = None
        revenue_ensemble = revenue_from_prob(y_pred_ensemble, all_true[0])
        try:
            logloss_ensemble = log_loss(all_true[0], y_pred_ensemble, labels=[0,1])
        except:
            logloss_ensemble = None

        logger.info(f"[Ensamble] AUC: {auc_ensemble if auc_ensemble is not None else 'N/A'}")
        logger.info(f"[Ensamble] Revenue: {revenue_ensemble:.2f}")
        logger.info(f"[Ensamble] LogLoss: {logloss_ensemble if logloss_ensemble is not None else 'N/A'}")


    #### Final Modeling ####
    logger.info("Iniciando Modelado final...")

    if not DEBUG:
        X_final_train = pd.concat([X_train, X_test])
        y_final_train = np.concatenate([y_train, y_test])
    else:
        X_final_train = X_train_sampled
        y_final_train = y_train_sampled

    logger.info(f"X_train.shape antes de transformar: {X_final_train.shape}")

    # Transformamos target y features
    X_final_train, y_final_train = target_processor.transform(X_final_train, y_final_train)
    X_final_train = delta_lag_transformer.transform(X_final_train)
    logger.info(f"X_train.shape después de transformar: {X_final_train.shape}")

    n_seeds = len(seeds)
    y_preds = []
    
    logger.info(f"Entrenando y prediciendo con {n_seeds} seeds para ensamblado...")
    for i, seed in enumerate(seeds):
        best_params_seed = best_params.copy()
        best_params_seed["seed"] = seed

        logger.info(f"[Seed {seed}] Entrenando modelo final")
        train_final_dataset = lgb.Dataset(X_final_train, label=y_final_train)
        model = lgb.train(best_params_seed, train_final_dataset)
        y_pred = model.predict(X_eval_transformed)
        y_preds.append(y_pred)

    y_preds = np.stack(y_preds)
    y_pred_ensemble = np.mean(y_preds, axis=0)
    # Save ensemble predictions to a CSV in the experiment folder
    prediction_df = pd.DataFrame({
        "numero_de_cliente": X_eval_transformed.index if hasattr(X_eval_transformed, "index") else np.arange(len(y_pred_ensemble)),
        "predicted_probability": y_pred_ensemble
    })
    prediction_file = os.path.join(experiment_path, "ensemble_predictions.csv")
    prediction_df.to_csv(prediction_file, index=False)
    logger.info(f"Predicciones de ensamblado guardadas en: {prediction_file}")

    logger.info(f"Experimento completado en {(time.time() - start_time)/60:.2f} minutos")

