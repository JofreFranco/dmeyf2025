"""
Utilidades para carga de datos, cálculo de targets y transformaciones
"""
import os
import glob
import time


import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

from .data_dict import (
    EXCLUDE_COLS, 
    BANK_CAT_COLS, 
    MASTER_CAT_COLS, 
    VISA_CAT_COLS,
    ALL_CAT_COLS,
    TRAIN_MONTHS,
    TEST_MONTH,
    PESO_BAJA_2,
    PESO_BAJA_1,
    PESO_CONTINUA,
    GANANCIA_ACIERTO,
    COSTO_ESTIMULO,
    RANDOM_SEEDS
)
from .clase_ternaria import calcular_clase_ternaria, calcular_clase_binaria

# DATA

def cargar_datos(raw_data_path, target_data_path=None, recalcular=False):
    """
    Carga los datos desde archivo CSV y calcula la clase ternaria si es necesario.
    
    Parameters:
    -----------
    raw_data_path : str
        Ruta al archivo de datos crudos
    target_data_path : str, optional
        Ruta donde guardar/cargar los datos con target calculado
    recalcular : bool, default=False
        Si True, fuerza el recálculo del target aunque ya exista
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con los datos y las clases calculadas
    """
    
    # Si no se proporciona ruta de target, usar la misma ruta pero con sufijo _target
    if target_data_path is None:
        base_path = raw_data_path.rsplit('.', 1)[0]
        target_data_path = f"{base_path}_target.csv"
    
    # Verificar si ya existe el archivo con target calculado
    if not recalcular and os.path.exists(target_data_path):
        print("Cargando datos con target existente...")
        df = pd.read_csv(target_data_path)
    else:
        print("Calculando target desde datos crudos...")
        # Cargar datos crudos
        print(f"Cargando datos desde: {raw_data_path}")
        df = pd.read_csv(raw_data_path)
        print(f"Datos cargados: {df.shape[0]} filas y {df.shape[1]} columnas")

        # Calcular la clase ternaria
        print("Calculando clase ternaria...")
        df = calcular_clase_ternaria(df)
        
        # Calcular la clase binaria
        print("Calculando clase binaria...")
        df = calcular_clase_binaria(df)

        # Verificar la distribución de clases
        print("\nDistribución de la clase ternaria:")
        print(df['clase_ternaria'].value_counts())
        print("\nPorcentajes:")
        print(df['clase_ternaria'].value_counts(normalize=True) * 100)
        
        print("\nDistribución de la clase binaria:")
        print(df['clase_binaria'].value_counts())
        print("\nPorcentajes:")
        print(df['clase_binaria'].value_counts(normalize=True) * 100)
        
        # Guardar los datos con target calculado
        print(f"Guardando datos con target en: {target_data_path}")
        df.to_csv(target_data_path, index=False)
    
    return df

def calcular_pesos_clase(df):
    """
    Calcula los pesos de clase basado en la clase ternaria.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con la columna 'clase_ternaria'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con la nueva columna 'clase_peso'
    """
    df = df.copy()
    df['clase_peso'] = PESO_CONTINUA
    df.loc[df['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = PESO_BAJA_2
    df.loc[df['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = PESO_BAJA_1
    
    return df

def obtener_columnas_por_tipo(df):
    """
    Separa las columnas del DataFrame por tipo (bancarias, visa, master, categóricas, numéricas).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
        
    Returns:
    --------
    dict
        Diccionario con las listas de columnas separadas por tipo
    """
    # Identificar columnas Visa numéricas
    visa_num_cols = [col for col in df.columns if 'Visa' in col and col not in VISA_CAT_COLS]
    
    # Identificar columnas Master numéricas
    master_num_cols = [col for col in df.columns if 'Master' in col and col not in MASTER_CAT_COLS]
    
    # Identificar columnas bancarias numéricas (resto de columnas)
    bank_num_cols = [col for col in df.columns 
                     if col not in (BANK_CAT_COLS + MASTER_CAT_COLS + VISA_CAT_COLS + 
                                  visa_num_cols + master_num_cols + EXCLUDE_COLS)]
    
    return {
        'bank_cat_cols': BANK_CAT_COLS,
        'master_cat_cols': MASTER_CAT_COLS,
        'visa_cat_cols': VISA_CAT_COLS,
        'visa_num_cols': visa_num_cols,
        'master_num_cols': master_num_cols,
        'bank_num_cols': bank_num_cols,
        'all_cat_cols': ALL_CAT_COLS,
        'all_num_cols': visa_num_cols + master_num_cols + bank_num_cols
    }

def transformar_columnas_categoricas(df, cat_cols, metodo='astype'):
    """
    Transforma columnas a tipo categórico.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    cat_cols : list
        Lista de nombres de columnas categóricas
    metodo : str, default='astype'
        Método de transformación ('astype' o 'factorize')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con columnas transformadas
    """
    df = df.copy()
    
    for col in cat_cols:
        if col in df.columns:
            if metodo == 'astype':
                df[col] = df[col].astype('category')
            elif metodo == 'factorize':
                df[col], _ = pd.factorize(df[col])
                
    return df

def sample_dataset_estratificado(df, sample_ratio=1.0, debug_mode=False):
    """
    Realiza sampling estratificado del dataset, reduciendo solo la clase CONTINUA.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    sample_ratio : float, default=1.0
        Proporción de casos CONTINUA a mantener (0.0 a 1.0)
    debug_mode : bool, default=False
        Si True, mantiene solo 1000 casos CONTINUA independientemente de sample_ratio
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con sampling aplicado
    """
    if sample_ratio >= 1.0 and not debug_mode:
        return df.copy()
    
    df_sampled = df.copy()
    
    # Separar clases
    continua_mask = df_sampled['clase_ternaria'] == 'CONTINUA'
    other_classes = df_sampled[~continua_mask].copy()
    continua_cases = df_sampled[continua_mask].copy()
    
    # Determinar cuántos casos CONTINUA mantener
    if debug_mode:
        n_continua_keep = min(1000, len(continua_cases))
        print(f"🐛 DEBUG MODE: Manteniendo {n_continua_keep} casos CONTINUA")
    else:
        n_continua_keep = int(len(continua_cases) * sample_ratio)
        print(f"📊 SAMPLING: Manteniendo {n_continua_keep}/{len(continua_cases)} casos CONTINUA ({sample_ratio:.1%})")
    
    # Hacer sampling de casos CONTINUA
    if n_continua_keep < len(continua_cases):
        from sklearn.utils import resample
        continua_sampled = resample(
            continua_cases, 
            n_samples=n_continua_keep, 
            random_state=42,
            replace=False
        )
    else:
        continua_sampled = continua_cases
    
    # Combinar datasets
    df_final = pd.concat([other_classes, continua_sampled], ignore_index=True)
    
    print(f"✅ Dataset final: {len(df_final)} registros")
    print(f"   - BAJA+2: {(df_final['clase_ternaria'] == 'BAJA+2').sum()}")
    print(f"   - BAJA+1: {(df_final['clase_ternaria'] == 'BAJA+1').sum()}")  
    print(f"   - CONTINUA: {(df_final['clase_ternaria'] == 'CONTINUA').sum()}")
    
    return df_final

def load_and_prepare_data(raw_data_path, target_data_path=None, preprocessor=None):
    """
    Carga datos, calcula targets, aplica pesos y preprocessing.
    
    Parameters:
    -----------
    raw_data_path : str
        Ruta a datos crudos
    target_data_path : str, optional
        Ruta a datos con target calculado
    preprocessor : sklearn transformer, optional
        Preprocessor a aplicar
        
    Returns:
    --------
    pd.DataFrame
        DataFrame procesado y listo para usar
    """
    # Cargar datos
    df = cargar_datos(raw_data_path, target_data_path)
    
    # Calcular pesos
    df = calcular_pesos_clase(df)
    
    # Aplicar preprocessor
    if preprocessor is not None:
        preprocessor.fit(df)
        df = preprocessor.transform(df)
    
    return df

def split_train_test_eval(df, train_months=None, test_month=None, eval_month=None, 
                         sample_ratio=1.0, debug_mode=False):
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

# METRICS

def optimize_threshold(y_prob, y_true, weights=None, threshold_range=(0.01, 0.5), n_points=100):
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
    import numpy as np
    from .data_dict import GANANCIA_ACIERTO, COSTO_ESTIMULO
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

def lgb_auc_eval(y_pred, data):
    """
    Función de evaluación de AUC para LightGBM (callback).
    
    Parameters:
    -----------
    y_pred : array-like
        Predicciones del modelo (probabilidades)
    data : lightgbm.Dataset
        Dataset de LightGBM con etiquetas
        
    Returns:
    --------
    tuple
        (nombre_metrica, valor, is_higher_better)
    """
    from sklearn.metrics import roc_auc_score
    import numpy as np
    
    y_true = data.get_label()
    weights = data.get_weight()
    
    # Calcular AUC (con pesos si están disponibles)
    if weights is not None:
        auc = roc_auc_score(y_true, y_pred, sample_weight=weights)
    else:
        auc = roc_auc_score(y_true, y_pred)
    
    return 'auc_eval', auc, True

def lgb_gan_eval(y_pred, data):
    """
    Función de evaluación de ganancia para LightGBM (callback) usando ganancia_prob.
    
    Parameters:
    -----------
    y_pred : array-like
        Predicciones del modelo (probabilidades)
    data : lightgbm.Dataset
        Dataset de LightGBM con etiquetas
        
    Returns:
    --------
    tuple
        (nombre_metrica, valor, is_higher_better)
    """
    y_true = data.get_label()
    
    # Convertir a clase ternaria para ganancia_prob
    y_ternaria = np.where(y_true == 1, "BAJA+2", "CONTINUA")
    
    # Crear array 2D para ganancia_prob (columna 0: prob negativa, columna 1: prob positiva)
    y_hat_2d = np.column_stack([1 - y_pred, y_pred])
    
    ganancia_max = ganancia_prob(y_hat_2d, y_ternaria, prop=1, class_index=1, threshold=0.025)
    
    return 'gan_eval', ganancia_max, True

def calcular_ganancia(y_true, y_pred, y_weights=None, threshold=0.025):
    """
    Calcula la ganancia usando ganancia_prob (reemplaza la función anterior).
    
    Parameters:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_pred : array-like
        Probabilidades predichas
    y_weights : array-like, optional
        Pesos (no se usan en ganancia_prob, mantenido por compatibilidad)
    threshold : float, default=0.025
        Threshold para clasificar como positivo
        
    Returns:
    --------
    float
        Ganancia máxima
    """
    # Convertir etiquetas binarias a clase ternaria
    y_ternaria = np.where(y_true == 1, "BAJA+2", "CONTINUA")
    
    # Crear array 2D para ganancia_prob (columna 0: prob negativa, columna 1: prob positiva)
    y_hat_2d = np.column_stack([1 - y_pred, y_pred])
    
    return ganancia_prob(y_hat_2d, y_ternaria, prop=1, class_index=1, threshold=threshold)

def calcular_auc(y_true, y_pred):
    """
    Calcula el AUC (Area Under ROC Curve).
    
    Parameters:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_pred : array-like
        Probabilidades predichas
        
    Returns:
    --------
    float
        Valor de AUC
    """
    return roc_auc_score(y_true, y_pred)

def ganancia_prob(y_hat, y, prop=1, class_index=1, threshold=0.025):
    @np.vectorize
    def ganancia_row(predicted, actual, threshold=0.025):
        return (predicted >= threshold) * (GANANCIA_ACIERTO if actual == "BAJA+2" else -COSTO_ESTIMULO)

    return ganancia_row(y_hat[:,class_index], y).sum() / prop

# debug

def get_debug_filename(filename, debug_mode=False):
    """
    Agrega sufijo _DEBUG al nombre de archivo si está en modo debug.
    
    Parameters:
    -----------
    filename : str
        Nombre del archivo original
    debug_mode : bool
        Si está en modo debug
        
    Returns:
    --------
    str
        Nombre del archivo con sufijo si corresponde
    """
    if not debug_mode:
        return filename
    
    if '.' in filename:
        name, ext = filename.rsplit('.', 1)
        return f"{name}_DEBUG.{ext}"
    else:
        return f"{filename}_DEBUG"

# Optimización Bayesiana

def create_optuna_objective(hyperparameter_space, X_train, y_train, w_train, seed=None, n_folds=5, feval=None):
    """
    Crea función objetivo para optimización bayesiana con Optuna.
    
    Parameters:
    -----------
    hyperparameter_space : dict
        Espacio de búsqueda de hiperparámetros con formato:
        {'param_name': ('suggest_type', min_val, max_val), ...}
    X_train : array-like
        Features de entrenamiento
    y_train : array-like
        Target de entrenamiento
    w_train : array-like
        Pesos de entrenamiento
    seed : int, optional
        Semilla aleatoria
    n_folds : int, default=5
        Número de folds para CV
    feval : function, optional
        Función de evaluación para LightGBM. Si None, usa lgb_auc_eval por defecto
        
    Returns:
    --------
    function
        Función objetivo para Optuna
    """
    if seed is None:
        seed = RANDOM_SEEDS[0]
    
    # Usar función de evaluación por defecto si no se proporciona
    if feval is None:
        feval = lgb_auc_eval
    
    # Determinar nombre de métrica basado en la función de evaluación
    metric_name = feval.__name__.replace('lgb_', '').replace('_eval', '_eval')
    
    def objective(trial):
        trial_start_time = time.time()
        
        # Construir parámetros basados en el espacio de búsqueda
        params = {
            'objective': 'binary',
            'metric': 'custom',
            'boosting_type': 'gbdt',
            'first_metric_only': True,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'max_bin': 31,
            'seed': seed,
            'verbose': -1,
            'num_threads': 10
        }
        
        # Agregar hiperparámetros del espacio de búsqueda
        for param_name, (suggest_type, min_val, max_val) in hyperparameter_space.items():
            if suggest_type == 'int':
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
            elif suggest_type == 'float':
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            elif suggest_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, min_val)  # min_val es la lista de opciones
        
        # Preparar datos para CV
        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
        
        # Cross-validation con Early Stopping
        cv_results = lgb.cv(
            params,
            train_data,
            num_boost_round=2000,
            feval=feval,
            stratified=True,
            nfold=n_folds,
            seed=seed,
            return_cvbooster=False,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        
        # Obtener mejor valor de la métrica
        metric_key = f'valid {metric_name}-mean'
        max_value = max(cv_results[metric_key])
        best_iter = cv_results[metric_key].index(max_value) + 1
        
        # Guardar mejor iteración y duración
        trial.set_user_attr("best_iter", best_iter)
        
        print(f"✅ Trial {trial.number}: {metric_name}={max_value:.4f}, iter={best_iter}")
        
        return max_value
    
    return objective


def optimize_hyperparameters_with_optuna(hyperparameter_space, X_train, y_train, w_train, 
                                       n_trials=100, seed=None, n_folds=5, study_name="optimization",
                                       working_dir=None, feval=None):
    """
    Optimiza hiperparámetros usando Optuna y guarda todos los trials en CSV.
    
    Parameters:
    -----------
    hyperparameter_space : dict
        Espacio de búsqueda
    X_train, y_train, w_train : array-like
        Datos de entrenamiento
    n_trials : int
        Número de trials
    seed : int, optional
        Semilla aleatoria
    n_folds : int
        Número de folds para CV
    study_name : str
        Nombre del estudio
    working_dir : str, optional
        Directorio donde guardar archivos. Si None, usa directorio actual
    feval : function, optional
        Función de evaluación para LightGBM. Si None, usa lgb_auc_eval por defecto
        
    Returns:
    --------
    dict
        Mejores hiperparámetros
    """
    import os
    import pandas as pd
    import json
    
    if working_dir is None:
        working_dir = os.getcwd()
    
    # Usar función de evaluación por defecto si no se proporciona
    if feval is None:
        feval = lgb_auc_eval
    
    # Determinar nombre y tipo de métrica
    metric_name = feval.__name__.replace('lgb_', '').replace('_eval', '_eval')
    is_percentage_metric = 'auc' in metric_name.lower()
    
    # Crear función objetivo
    objective = create_optuna_objective(hyperparameter_space, X_train, y_train, w_train, seed, n_folds, feval)
    
    # Crear estudio con pruner para parar trials malos temprano
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Esperar 5 trials antes de empezar pruning
        n_warmup_steps=10    # Esperar 10 pasos dentro de cada trial
    )
    
    study = optuna.create_study(
        direction="maximize", 
        study_name=study_name,
        pruner=pruner
    )
    
    print(f"🚀 Iniciando optimización con early stopping y pruning inteligente...")
    print(f"⏰ Timeout por trial: 20 minutos")
    print(f"🛑 Early stopping: 50 rondas sin mejora")
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Guardar todos los trials en CSV
    trials_data = []
    for trial in study.trials:
        trial_dict = {
            'trial_number': trial.number,
            'value': trial.value,
            'state': trial.state.name,
            'best_iter': trial.user_attrs.get('best_iter', None),
            'duration_minutes': trial.user_attrs.get('duration_minutes', None)
        }
        # Agregar hiperparámetros
        trial_dict.update(trial.params)
        trials_data.append(trial_dict)
    
    trials_df = pd.DataFrame(trials_data)
    trials_csv_path = os.path.join(working_dir, f"{study_name}_trials.csv")
    trials_df.to_csv(trials_csv_path, index=False)
    print(f"💾 Trials guardados en: {trials_csv_path}")
    
    # Construir mejores parámetros completos
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'binary',
        'metric': metric_name,
        'boosting_type': 'gbdt',
        'max_bin': 31,
        'num_threads': 10,
        'verbose': 0,
        'num_boost_round': study.best_trial.user_attrs['best_iter']
    })
    
    # Guardar mejores hiperparámetros en JSON
    best_params_path = os.path.join(working_dir, f"{study_name}_best_params.json")
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    print(f"💾 Mejores hiperparámetros guardados en: {best_params_path}")
    
    print(f"✅ Optimización completada!")
    # Formatear valor según el tipo de métrica
    if is_percentage_metric:
        print(f"🏆 Mejor {metric_name.upper()}: {study.best_value:.4f}")
    else:
        print(f"🏆 Mejor {metric_name}: {study.best_value:,.0f}")
    print(f"🔧 Mejores parámetros:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    return best_params

def plot_feature_importance(model, feature_names=None, top_features=30, 
                          experiment_name="experiment", working_dir=".", debug_mode=False):
    """
    Genera y guarda un gráfico de importancia de features de un modelo LightGBM.
    
    Parameters:
    -----------
    model : lightgbm.Booster
        Modelo LightGBM entrenado
    feature_names : list, optional
        Nombres de las features. Si None, usa números
    top_features : int, default=30
        Número de features más importantes a mostrar
    experiment_name : str
        Nombre del experimento para el archivo
    working_dir : str
        Directorio donde guardar la imagen
    debug_mode : bool
        Si True, agrega _DEBUG al nombre del archivo
        
    Returns:
    --------
    str
        Ruta del archivo de imagen guardado
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Backend sin display para servidores
    import seaborn as sns
    import pandas as pd
    import os
    
    # Obtener importancia de features
    importance = model.feature_importance(importance_type='gain')
    
    # Crear nombres de features si no se proporcionan
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importance))]
    
    # Crear DataFrame con importancia
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importance)],
        'importance': importance
    })
    
    # Ordenar por importancia y tomar las top features
    importance_df = importance_df.nlargest(top_features, 'importance')
    
    # Configurar el gráfico (tamaño ajustado para 30 features)
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    
    # Crear barplot horizontal
    ax = sns.barplot(
        data=importance_df, 
        y='feature', 
        x='importance',
        palette='viridis'
    )
    
    # Personalizar el gráfico
    plt.title(f'Feature Importance - {experiment_name.title()}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Importance (Gain)', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    
    # Ajustar tamaño de fuente para acomodar más features
    plt.setp(ax.get_yticklabels(), fontsize=9)
    plt.setp(ax.get_xticklabels(), fontsize=10)
    
    # Agregar valores en las barras
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        ax.text(row['importance'] + max(importance_df['importance']) * 0.01, 
                i, f'{row["importance"]:.0f}', 
                va='center', fontsize=8, fontweight='bold')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar imagen siguiendo convención de nombres
    image_filename = get_debug_filename(f"{experiment_name}_feature_importance.png", debug_mode)
    image_path = os.path.join(working_dir, image_filename)
    
    plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Cerrar para liberar memoria
    
    print(f"📊 Feature importance guardado: {image_filename}")
    
    return image_path


def prob_to_prediction(input_csv_path, threshold=0.025, output_csv_path=None):
    """
    Convierte probabilidades en predicciones binarias aplicando un threshold.
    
    Parameters:
    -----------
    input_csv_path : str
        Ruta al archivo CSV con probabilidades
    threshold : float, default=0.025
        Threshold para convertir probabilidad en predicción (>=threshold = 1, <threshold = 0)
    output_csv_path : str, optional
        Ruta del archivo de salida. Si None, se genera automáticamente agregando "_prediction"
        
    Returns:
    --------
    str
        Ruta del archivo generado
    """

    
    # Leer archivo de probabilidades
    df = pd.read_csv(input_csv_path)
    
    if 'probabilidad' not in df.columns:
        raise ValueError(f"El archivo {input_csv_path} debe contener una columna 'probabilidad'")
    
    # Crear predicciones binarias
    df['Predicted'] = (df['probabilidad'] >= threshold).astype(int)
    
    # Generar nombre del archivo de salida si no se especifica
    if output_csv_path is None:
        base_path = input_csv_path.rsplit('.', 1)[0]
        output_csv_path = f"{base_path}_prediction.csv"
    
    # Guardar archivo con predicciones
    df_output = df[['numero_de_cliente', 'Predicted']].copy()
    df_output.to_csv(output_csv_path, index=False)
    
    num_predictions_1 = df['Predicted'].sum()
    total_predictions = len(df)
    percentage_1 = (num_predictions_1 / total_predictions) * 100
    
    print(f"✅ Predicciones generadas: {output_csv_path}")
    print(f"   Threshold usado: {threshold}")
    print(f"   Predicciones = 1: {num_predictions_1:,} ({percentage_1:.2f}%)")
    print(f"   Predicciones = 0: {total_predictions - num_predictions_1:,} ({100-percentage_1:.2f}%)")
    
    return output_csv_path


def process_experiment_predictions(experiment_dir, threshold=0.025, pattern_suffix=None, debug_mode=False):
    """
    Procesa todos los archivos de predicciones de un experimento y genera archivos de predicción.
    
    Parameters:
    -----------
    experiment_dir : str
        Directorio del experimento
    threshold : float, default=0.025
        Threshold para las predicciones
    pattern_suffix : str, default="_ensemble"
        Sufijo de los archivos a procesar (ej: "_ensemble", "_537919", etc.)
        Si es None, procesa todos los archivos .csv que contengan probabilidades
    debug_mode : bool, default=False
        Si está en modo debug
        
    Returns:
    --------
    list
        Lista de archivos de predicción generados
    """

    # Escapar caracteres especiales en el directorio usando glob.escape()
    escaped_dir = glob.escape(experiment_dir)
    prediction_files = []
    
    if pattern_suffix is None:
        # Buscar todos los archivos CSV
        pattern = os.path.join(escaped_dir, "*.csv")
        csv_files = glob.glob(pattern)
        print(f"🔍 Patrón de búsqueda: {pattern}")
        print(f"🔍 Archivos encontrados: {len(csv_files)}")
    else:
        # Buscar archivos con el patrón específico
        pattern = os.path.join(escaped_dir, f"*{pattern_suffix}.csv")
        csv_files = glob.glob(pattern)
        print(f"🔍 Patrón de búsqueda: {pattern}")
        print(f"🔍 Archivos encontrados: {len(csv_files)}")
    
    print(f"🔍 Procesando archivos de predicciones en: {experiment_dir}")
    print(f"   Threshold: {threshold}")
    print(f"   Patrón: *{pattern_suffix}.csv" if pattern_suffix else "   Patrón: *.csv")
    if csv_files:
        for csv_file in csv_files:
            try:
                # Verificar si el archivo contiene probabilidades
                df_test = pd.read_csv(csv_file, nrows=1)
                
                if 'probabilidad' in df_test.columns:
                    pred_file = prob_to_prediction(csv_file, threshold=threshold)
                    prediction_files.append(pred_file)
                    print(f"   ✅ {os.path.basename(csv_file)} → {os.path.basename(pred_file)}")
                else:
                    print(f"   ⏭️  {os.path.basename(csv_file)} (no contiene columna 'probabilidad')")
                    
            except Exception as e:
                print(f"   ❌ Error procesando {os.path.basename(csv_file)}: {e}")
        
        print(f"\n🎉 Procesamiento completado: {len(prediction_files)} archivos de predicción generados")
        return prediction_files
    else:
        print(f"❌ No se encontraron archivos CSV en {experiment_dir}")
        return []
