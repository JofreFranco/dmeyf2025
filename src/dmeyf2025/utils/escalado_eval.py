import logging
logger = logging.getLogger(__name__)

def scale_params(best_params, X_train, X_train_sampled):
    """
    Escala los parámetros de la mejor trial para el tamaño de la muestra de entrenamiento
    """
    logger.info(f"min_data_in_leaf: {best_params['min_data_in_leaf']}")
    best_params["min_data_in_leaf"] = int(best_params["min_data_in_leaf"]/(len(X_train_sampled)/len(X_train)))
    logger.info(f"factor de escalado: {round(len(X_train_sampled)/len(X_train), 2)}")
    logger.info(f"min_data_in_leaf escalado: {best_params['min_data_in_leaf']}")

    return best_params