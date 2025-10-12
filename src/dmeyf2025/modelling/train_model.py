import lightgbm as lgb
import logging
import os
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def train_model(X_train, y_train, params):

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data)

    return model

def predict_ensemble_model(models, X_eval):
    predictions = pd.DataFrame()
    predictions["numero_de_cliente"] = X_eval.index
    for n, model in enumerate(models):
        y_pred = model.predict(X_eval)
        predictions[f"pred_{n}"] = y_pred
    predictions["pred_ensemble"] = predictions.drop(columns=["numero_de_cliente"]).mean(axis=1)
    predictions = predictions.sort_values(by="pred_ensemble", ascending=False)
    return predictions

def train_models(X_train, y_train, X_eval, params, seeds, experiment_path=None):
    logger.info(f"Training dataset shape: {X_train.shape}")
    logger.info(f"Evaluating dataset shape: {X_eval.shape}")
    models = []
    for seed in seeds:
        params["seed"] = seed
        logger.info(f"Training final model with seed: {seed}")
        model = train_model(X_train, y_train, params)
        models.append(model)
    predictions = predict_ensemble_model(models, X_eval)
    return predictions, models

def prob_to_sends(experiment_config,predictions, n_sends, name="ensemble_predictions"):
    experiment_path = f"{experiment_config['experiments_path']}/{experiment_config['experiment_folder']}"
    ones = np.ones(n_sends, dtype=int)
    zeros = np.zeros(len(predictions)-n_sends, dtype=int)
    sends = np.concatenate([ones, zeros])
    predictions["predicted"] = sends
    predictions[["numero_de_cliente", "predicted"]].to_csv(f"{experiment_path}/{experiment_config["experiment_folder"]}_{name}.csv", index=False)