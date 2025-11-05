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

def lgb_gan_eval(y_pred, data):
    weight = data.get_weight()

    ganancia = np.where(weight == 1.00002, GANANCIA_ACIERTO, 0) - np.where(weight < 1.00002, COSTO_ESTIMULO, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)
    sends = np.argmax(ganancia)
    return 'gan', np.mean(ganancia[sends-1000:sends+1000]) , True
def sends_optimization(y_pred, weight):

    ganancia = np.where(weight == 1.00002, GANANCIA_ACIERTO, 0) - np.where(weight < 1.00002, COSTO_ESTIMULO, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)
    return np.argmax(ganancia), np.max(ganancia)
