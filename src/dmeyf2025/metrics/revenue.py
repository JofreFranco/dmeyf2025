import numpy as np
from dmeyf2025.utils.data_dict import GANANCIA_ACIERTO, COSTO_ESTIMULO

class Feval:
    def __init__(self):
        self.best_ks = []

    def __call__(self, y_pred, dataset):
        y_true = dataset.get_label()
        sorted_idx = np.argsort(-y_pred)
        y_true_sorted = y_true[sorted_idx]
        tp_cum = np.cumsum(y_true_sorted)
        fp_cum = np.cumsum(1 - y_true_sorted)

        gain = tp_cum * 780000 + fp_cum * -20000
        best_k = np.argmax(gain)
        best_gain = gain[best_k]

        self.best_ks.append(int(best_k))
        return "gan", best_gain, True


def revenue_from_prob(y_pred, y_true, n_envios=10000):
    """
    Calcula la ganancia esperada enviando estímulos a los N clientes con mayor probabilidad.
    
    Parameters:
    -----------
    y_pred : array-like
        Probabilidades predichas para la clase positiva
    y_true : array-like
        Etiquetas verdaderas (strings: "CONTINUA", "BAJA+1", "BAJA+2")
    n_envios : int, default=11000
        Número de envíos (clientes a los que se les envía estímulo con mayor probabilidad)
        
    Returns:
    --------
    float
        Ganancia total calculada
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Convertir etiquetas ternarias a binarias (solo BAJA+2 es positiva)
    y_true_binary = (y_true == "BAJA+2").astype(int)
    
    # Seleccionar índices de los n_envios mayores
    if n_envios >= len(y_pred):
        y_pred_binary = np.ones_like(y_pred)
    else:
        orden = np.argsort(-y_pred)  # orden descendente
        y_pred_binary = np.zeros_like(y_pred, dtype=int)
        y_pred_binary[orden[:n_envios]] = 1

    # Calcular métricas
    tp = np.sum(((y_true_binary == 1) & (y_pred_binary == 1)).astype(int))
    fp = np.sum(((y_true_binary == 0) & (y_pred_binary == 1)).astype(int))

    # Calcular ganancia
    ganancia = tp * GANANCIA_ACIERTO - fp * COSTO_ESTIMULO
    return ganancia

def lgb_gan_eval2(y_pred, dataset):
    y_true = dataset.get_label()
    n = len(y_true)

    # Ordeno de mayor a menor predicción
    sorted_idx = np.argsort(-y_pred)
    y_true_sorted = y_true[sorted_idx]

    # Ganancia acumulada si marco como positivo los primeros k
    # TP = +780000, FP = -20000
    gain_per_TP = 780000
    loss_per_FP = -20000

    tp_cum = np.cumsum(y_true_sorted)
    fp_cum = np.cumsum(1 - y_true_sorted)

    gain = tp_cum * gain_per_TP + fp_cum * loss_per_FP

    # Busco el k que maximiza la ganancia
    best_k = np.argmax(gain)
    best_gain = gain[best_k]

    # LightGBM espera que devuelvas:
    # (nombre, valor, is_higher_better)
    return "gan", best_gain, True

def lgb_gan_eval(y_pred, data):

    """
    Función de evaluación personalizada para LightGBM que calcula ganancia.
    Solo BAJA+2 es considerada como clase positiva.
    
    Parameters:
    -----------
    y_pred : array-like
        Predicciones del modelo (probabilidades para clase 1)
    data : lgb.Dataset
        Dataset de LightGBM con las etiquetas verdaderas
        
    Returns:
    --------
    tuple
        (eval_name, eval_result, is_higher_better)
    """

    y_true = data.get_label()

    # Convertir etiquetas binarias (0/1) a formato ternario para ganancia_prob
    # Solo BAJA+2 es considerada positiva (clase 1), BAJA+1 ahora es CONTINUA
    y_ternaria = ["CONTINUA" if label == 0 else "BAJA+2" for label in y_true]
    
    # Calcular ganancia usando las probabilidades directamente
    ganancia = revenue_from_prob(y_pred, y_ternaria, n_envios=4000)

    return 'gan', ganancia, True

def sends_optimization(y_pred, y_true, min_sends, max_sends, step=500):
    """
    Función que optimiza la cantidad de envíos para maximizar la ganancia.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Convertir etiquetas binarias (0/1) a formato ternario para ganancia_prob
    y_ternaria = ["CONTINUA" if label == 0 else "BAJA+2" for label in y_true]
    max_ganancia = -np.inf
    # Calcular ganancia usando las probabilidades directamente
    for n_sends in range(min_sends, max_sends, step):
        ganancia = revenue_from_prob(y_pred, y_ternaria, n_sends)
        if ganancia > max_ganancia:
            max_ganancia = ganancia
            best_n_sends = n_sends
    return best_n_sends, max_ganancia
