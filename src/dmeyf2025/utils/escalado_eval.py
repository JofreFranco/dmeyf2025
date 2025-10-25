###### Esto no anda, está para tener el código por las dudas
"""
# Scale median
    X_scaled = scale(X, strategy="median")
    percentile_transformer = PercentileTransformer(variables=FINANCIAL_COLS)
    X_transformed = percentile_transformer.fit_transform(X_scaled)
    X_transformed.set_index("numero_de_cliente", inplace=True)

    X_eval = X_transformed[X_transformed["foto_mes"].isin([experiment_config['eval_month']])]
    X_eval = X_eval.drop(columns=["label"])
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
    percentile_transformer = PercentileTransformer(variables=FINANCIAL_COLS)
    X_transformed = percentile_transformer.fit_transform(X_scaled)
    X_transformed.set_index("numero_de_cliente", inplace=True)

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
    predictions[["numero_de_cliente", "predicted"]].to_csv(f"{experiment_path}/{experiment_config["experiment_folder"]}_ensemble_predictions_scaled_mean.csv", index=False)"""