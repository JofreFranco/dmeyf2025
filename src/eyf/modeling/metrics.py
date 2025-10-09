"""
Funciones para métricas y evaluación de modelos
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from ..utils.data_dict import COSTO_ESTIMULO, GANANCIA_ACIERTO


def optimize_threshold(y_prob, y_true, weights=None, threshold_range=(0.01, 0.10), n_points=100):
    """
    Optimiza el threshold para maximizar la ganancia.
    
    Parameters:
    -----------
    y_prob : array-like
        Probabilidades predichas
    y_true : array-like
        Etiquetas verdaderas (0/1)
    weights : array-like, optional
        Pesos de las muestras
    threshold_range : tuple, optional
        Rango de thresholds a probar (min, max)
    n_points : int, optional
        Número de puntos a evaluar en el rango
        
    Returns:
    --------
    dict
        Diccionario con threshold óptimo y métricas
    """
    if weights is None:
        weights = np.ones(len(y_true))
    
    # Generar thresholds a probar
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_points)
    
    best_threshold = None
    best_gain = float('-inf')
    results = []
    
    print(f"🔍 Probando {n_points} thresholds en rango [{threshold_range[0]:.3f}, {threshold_range[1]:.3f}]")
    
    for threshold in thresholds:
        # Aplicar threshold
        y_pred_binary = (y_prob >= threshold).astype(int)
        #print((y_true == 1) & (y_pred_binary == 1))
        
        # Calcular métricas ponderadas
        tp = np.sum(((y_true == 1) & (y_pred_binary == 1)).astype(int) * weights)
        fp = np.sum(((y_true == 0) & (y_pred_binary == 1)).astype(int) * weights)
        tn = np.sum(((y_true == 0) & (y_pred_binary == 0)).astype(int) * weights)
        fn = np.sum(((y_true == 1) & (y_pred_binary == 0)).astype(int) * weights)
        
        # Calcular ganancia
        ganancia = tp * GANANCIA_ACIERTO - fp * COSTO_ESTIMULO
        
        # Calcular otras métricas
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'ganancia': ganancia,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        if ganancia > best_gain:
            best_gain = ganancia
            best_threshold = threshold
    
    best_result = next(r for r in results if r['threshold'] == best_threshold)
    
    print(f"✅ Threshold óptimo: {best_threshold:.4f}")
    print(f"💰 Ganancia máxima: {best_gain:,.0f}")
    print(f"📊 Precision: {best_result['precision']:.4f}, Recall: {best_result['recall']:.4f}")
    
    return {
        'optimal_threshold': best_threshold,
        'best_gain': best_gain,
        'best_metrics': best_result,
        'all_results': results
    }


def calcular_ganancia(y_true, y_pred, y_weights=None, threshold=0.025):
    """
    Calcula la ganancia usando predicciones binarias.
    
    Parameters:
    -----------
    y_true : array-like
        Etiquetas verdaderas (0/1)
    y_pred : array-like  
        Predicciones del modelo (0/1 o probabilidades)
    y_weights : array-like, optional
        Pesos de las muestras
    threshold : float, default=0.025
        Threshold para convertir probabilidades en predicciones binarias
        
    Returns:
    --------
    float
        Ganancia calculada
    """
    # Si y_pred son probabilidades, convertir a binario
    if y_pred.max() <= 1.0 and y_pred.min() >= 0.0:
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)
    
    if y_weights is None:
        y_weights = np.ones(len(y_true))
    
    # Calcular matriz de confusión ponderada
    tp = np.sum((y_true == 1) & (y_pred_binary == 1) * y_weights)
    fp = np.sum((y_true == 0) & (y_pred_binary == 1) * y_weights)
    
    # Calcular ganancia
    ganancia = tp * GANANCIA_ACIERTO - fp * COSTO_ESTIMULO
    
    return ganancia


def calcular_auc(y_true, y_pred):
    """
    Calcula el AUC-ROC de las predicciones.
    
    Parameters:
    -----------
    y_true : array-like
        Etiquetas verdaderas (0/1)
    y_pred : array-like
        Predicciones del modelo (probabilidades)
        
    Returns:
    --------
    float
        Valor de AUC-ROC
    """
    return roc_auc_score(y_true, y_pred)


def ganancia_prob(y_hat, y, prop=1, class_index=1, threshold=0.025):
    """
    Calcula la ganancia esperada basada en probabilidades.
    
    Parameters:
    -----------
    y_hat : array-like
        Array 2D con probabilidades [prob_class_0, prob_class_1]
    y : array-like
        Etiquetas verdaderas (strings: "CONTINUA", "BAJA+1", "BAJA+2")
    prop : float, default=1
        Proporción de casos a considerar
    class_index : int, default=1
        Índice de la clase positiva en y_hat
    threshold : float, default=0.025
        Threshold para clasificación
        
    Returns:
    --------
    float
        Ganancia total calculada
    """
    y_hat = np.array(y_hat)
    y = np.array(y)
    
    # Convertir probabilidades a predicciones binarias
    prob_positiva = y_hat[:, class_index] if y_hat.ndim > 1 else y_hat
    predicciones = (prob_positiva >= threshold).astype(int)
    
    # Convertir etiquetas a binario (solo BAJA+2 es positivo)
    y_binario = np.where(y == "BAJA+2", 1, 0)
    
    # Calcular matriz de confusión
    tp = np.sum((y_binario == 1) & (predicciones == 1))
    fp = np.sum((y_binario == 0) & (predicciones == 1))
    
    # Calcular ganancia
    ganancia = tp * GANANCIA_ACIERTO - fp * COSTO_ESTIMULO
    
    return ganancia * prop
