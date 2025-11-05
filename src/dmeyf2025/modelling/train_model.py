import lightgbm as lgb
import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def train_model(X_train, y_train, w_train, params):
    fixed_params = {
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
        }
    params = {**fixed_params, **params}
    
    # Calcular valores absolutos a partir de valores relativos si existen
    n_train = len(X_train)
    if 'rel_min_data_in_leaf' in params:
        rel_min_data_in_leaf = params.pop('rel_min_data_in_leaf')
        params['min_data_in_leaf'] = max(1, int(rel_min_data_in_leaf * n_train))
        logger.info(f"Calculando min_data_in_leaf desde rel_min_data_in_leaf: {rel_min_data_in_leaf:.6f} -> {params['min_data_in_leaf']}")
    
    if 'rel_num_leaves' in params:
        rel_num_leaves = params.pop('rel_num_leaves')
        min_data_in_leaf = params.get('min_data_in_leaf', 1)
        params['num_leaves'] = max(2, int(2 + rel_num_leaves * n_train / min_data_in_leaf))
        logger.info(f"Calculando num_leaves desde rel_num_leaves: {rel_num_leaves:.6f} -> {params['num_leaves']}")
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    model = lgb.train(params, train_data)

    return model

def predict_ensemble_model(models, X_eval):
    predictions = pd.DataFrame()
    predictions["numero_de_cliente"] = X_eval["numero_de_cliente"]
    logger.info(f"Predicting ensemble model with {len(models)} models")
    for n, model in enumerate(models):
        y_pred = model.predict(X_eval)
        predictions[f"pred_{n}"] = y_pred
    predictions["pred_ensemble"] = predictions.drop(columns=["numero_de_cliente"]).mean(axis=1)
    return predictions

def train_models(X_train, y_train, X_eval, params, seeds, w_train, experiment_path=None):

    logger.info(f"Training dataset shape: {X_train.shape}")
    logger.info(f"Evaluating dataset shape: {X_eval.shape}")
    models = []

    if "weight" in X_train.columns:
        logger.warning("Weight column found in X_train, removing it")
        w_train = X_train["weight"]
        X_train = X_train.drop(columns=["weight"])
    if "weight" in X_eval.columns:
        logger.warning("Weight column found in X_eval, removing it")
        w_eval = X_eval["weight"]
        X_eval = X_eval.drop(columns=["weight"])
    for seed in seeds:
        params_copy = params.copy()
        params_copy["seed"] = seed
        logger.info(f"Training final model with seed: {seed}")
        model = train_model(X_train, y_train, w_train, params_copy)
        models.append(model)
    predictions = predict_ensemble_model(models, X_eval)
    return predictions, models

def prob_to_sends(experiment_config, predictions, n_sends, name="ensemble_predictions"):
    predictions = predictions.sort_values(by="pred_ensemble", ascending=False)
    experiment_path = experiment_config['experiment_dir']
    ones = np.ones(n_sends, dtype=int)
    zeros = np.zeros(len(predictions)-n_sends, dtype=int)
    sends = np.concatenate([ones, zeros])
    predictions["predicted"] = sends
    output_file = experiment_path / f"{experiment_config['experiment_folder']}_{name}.csv"
    predictions[["numero_de_cliente", "predicted"]].to_csv(output_file, index=False)
    logger.info(f"Archivo guardado: {output_file}")
    return str(output_file)