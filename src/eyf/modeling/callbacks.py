"""
Callbacks para LightGBM
"""
from sklearn.metrics import roc_auc_score

from .metrics import ganancia_prob


def lgb_auc_eval(y_pred, data):
    """
    Función de evaluación personalizada para LightGBM que calcula AUC.
    
    Parameters:
    -----------
    y_pred : array-like
        Predicciones del modelo
    data : lgb.Dataset
        Dataset de LightGBM con las etiquetas verdaderas
        
    Returns:
    --------
    tuple
        (eval_name, eval_result, is_higher_better)
    """
    y_true = data.get_label()
    
    # Calcular AUC
    auc = roc_auc_score(y_true, y_pred)
    
    return 'auc', auc, True


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
    
    # Convertir probabilidades a formato 2D para ganancia_prob
    y_pred_2d = [[1-p, p] for p in y_pred]
    
    # Convertir etiquetas binarias (0/1) a formato ternario para ganancia_prob
    # Solo BAJA+2 es considerada positiva (clase 1), BAJA+1 ahora es CONTINUA
    y_ternaria = ["CONTINUA" if label == 0 else "BAJA+2" for label in y_true]
    
    # Calcular ganancia
    ganancia = ganancia_prob(y_pred_2d, y_ternaria, prop=1, class_index=1, threshold=0.025)
    
    return 'ganancia', ganancia, True
