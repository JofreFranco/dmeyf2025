"""
Funciones para división de datos en train/test/eval
"""
import pandas as pd
import numpy as np

from ..utils.data_dict import EXCLUDE_COLS
from .preprocessing import sample_dataset_estratificado


def split_train_test_eval(df, train_months=None, test_month=None, eval_month=None, sample_ratio=1.0, debug_mode=False):
    """
    Divide DataFrame en train, test y eval según foto_mes y aplica sampling al conjunto de entrenamiento.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con datos
    train_months : list, optional
        Meses para entrenamiento
    test_month : int, optional
        Mes para test
    eval_month : int, optional
        Mes para evaluación
    sample_ratio : float, optional
        Proporción de casos CONTINUA a mantener en entrenamiento (1.0 = todos)
    debug_mode : bool, optional
        Si True, aplica sampling para debug (1000 casos CONTINUA)
        
    Returns:
    --------
    dict
        Diccionario con las siguientes claves:
        - 'train': (X_train, y_train, w_train) - datos con sampling aplicado
        - 'train_full': (X_train_full, y_train_full, w_train_full) - datos sin sampling
        - 'test': (X_test, y_test, w_test) - datos de test
        - 'eval': (X_eval, customer_id_eval) - datos de evaluación
    """
    # Usar valores por defecto si no se especifican
    if train_months is None:
        train_months = [202101, 202102, 202103]
    if test_month is None:
        test_month = 202104
    if eval_month is None:
        eval_month = 202106
    
    # Dividir por mes
    train_data = df[df['foto_mes'].isin(train_months)].copy()
    test_data = df[df['foto_mes'] == test_month].copy()
    eval_data = df[df['foto_mes'] == eval_month].copy()
    
    # Identificar columnas de features
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
    
    # Extraer datos de test y eval (sin cambios)
    X_test = test_data[feature_cols].values
    X_eval = eval_data[feature_cols].values
    y_test = test_data['clase_binaria'].values
    w_test = test_data['clase_peso'].values
    customer_id_eval = eval_data['numero_de_cliente'].values
    
    # Extraer datos de entrenamiento SIN sampling (completos)
    X_train_full = train_data[feature_cols].values
    y_train_full = train_data['clase_binaria'].values
    w_train_full = train_data['clase_peso'].values
    
    # Aplicar sampling al conjunto de entrenamiento si es necesario
    if debug_mode or sample_ratio < 1.0:
        print(f"🎲 Aplicando sampling al conjunto de entrenamiento...")
        train_data_sampled = sample_dataset_estratificado(train_data, sample_ratio, debug_mode)
        X_train = train_data_sampled[feature_cols].values
        y_train = train_data_sampled['clase_binaria'].values
        w_train = train_data_sampled['clase_peso'].values
    else:
        print(f"📊 Usando conjunto de entrenamiento completo (sin sampling)")
        X_train = X_train_full
        y_train = y_train_full
        w_train = w_train_full
    if debug_mode:
        test_data_sampled = sample_dataset_estratificado(test_data, sample_ratio, debug_mode)
        X_test = test_data_sampled[feature_cols].values
        y_test = test_data_sampled['clase_binaria'].values
        w_test = test_data_sampled['clase_peso'].values
        eval_data_sampled = sample_dataset_estratificado(eval_data, sample_ratio, debug_mode)
        X_eval = eval_data_sampled[feature_cols].values
        customer_id_eval = eval_data_sampled['numero_de_cliente'].values
    return {
        'train': (X_train, y_train, w_train),
        'train_full': (X_train_full, y_train_full, w_train_full),
        'test': (X_test, y_test, w_test),
        'eval': (X_eval, customer_id_eval)
    }


def split_train_test_eval_legacy(df, train_months=None, test_month=None, eval_month=None):
    """
    Versión legacy de split_train_test_eval para compatibilidad hacia atrás.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con datos
    train_months : list, optional
        Meses para entrenamiento
    test_month : int, optional
        Mes para test
    eval_month : int, optional
        Mes para evaluación
        
    Returns:
    --------
    tuple
        (X_train, y_train, w_train, X_test, y_test, w_test, X_eval, customer_id_eval)
    """
    data_splits = split_train_test_eval(df, train_months, test_month, eval_month, 
                                       sample_ratio=1.0, debug_mode=False)
    
    X_train, y_train, w_train = data_splits['train']
    X_test, y_test, w_test = data_splits['test']
    X_eval, customer_id_eval = data_splits['eval']
    
    return X_train, y_train, w_train, X_test, y_test, w_test, X_eval, customer_id_eval
