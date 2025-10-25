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