import logging
import gc
import pandas as pd
import time
import lightgbm as lgb
import psutil
import os
import csv
import joblib
import numpy as np
from dmeyf2025.metrics.revenue import gan_eval
from dmeyf2025.processors.feature_processors import AddCanaritos

pd.set_option('display.max_columns', None)

def setup_logger(log_file="/home/martin232009/buckets/b1/experiment.log"):

    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # Evitar duplicados
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # ---- File handler ----
    try:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        print(f"[LOG] FileHandler creado OK en: {log_file}")
    except Exception as e:
        print(f"[ERROR] FileHandler falló: {e}")

    # ---- Console handler ----
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
logger = setup_logger()
def memory_gb(df: pd.DataFrame) -> float:
    return df.memory_usage().sum() / (1024 ** 3)

def apply_transformer(transformer, X, name: str, logger, VERBOSE=False, parallel=False, parallelize_by='foto_mes', n_jobs=-1):
    logger.info(f"[{name}] Iniciando…")

    start_mem = memory_gb(X)
    start_time = time.time()

    Xt = transformer.fit_transform(X, parallel=parallel, parallelize_by=parallelize_by, n_jobs=n_jobs)

    end_time = time.time()
    end_mem = memory_gb(Xt)

    n_rows, n_cols = Xt.shape

    logger.info(
        f"[{name}] Tiempo: {end_time - start_time:.2f}s | "
        f"Memoria antes: {start_mem:.3f} GB | "
        f"Memoria después: {end_mem:.3f} GB | "
        f"Diferencia: {end_mem - start_mem:+.3f} GB | "
        f"Shape: {n_rows:,} filas × {n_cols:,} columnas"
    )
    if VERBOSE:
        display(Xt.head())
        display(Xt.describe())
        logger.info(f"Nulos: {Xt.isna().astype(int).sum()}")
    gc.collect()
    return Xt

def train_model(train_set, params):
    """
    Entrena un modelo Z(uper)LightGBM (lgbm)
    Args:
        X_train (pd.DataFrame): Features de entrenamiento
        y_train (pd.Series): Variable objetivo de entrenamiento
        w_train (pd.Series): Weights
        params (dict): diccionario que debe tener:
            - 'semilla_primigenia'
            - 'min_data_in_leaf'
            - 'learning_rate'
            - 'canaritos': maneja el overfitting mediante canaritos, cuando detecta un árbol cuyo primer split es un canarito lo mata.
            - 'gradient_bound': bound para el gradiente es algo asi como un learning rate que va cambiando a medida que se va entrenando???.
    """
    lgb_params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "None",        # Para usar métrica custom
        "first_metric_only": False,
        "boost_from_average": True,
        "feature_pre_filter": False,
        "force_row_wise": True,
        "verbosity": -100,
        "seed": params["seed"],

        "max_bin": 31,
        "min_data_in_leaf": params["min_data_in_leaf"],

        "num_iterations": 9999,
        "num_leaves": 9999,
        "learning_rate": 1,

        "feature_fraction": params["feature_fraction"],

        # Hiperparámetros del Zuperlightgbm
        "canaritos": params["canaritos"],
        "gradient_bound": params["gradient_bound"],  
    }

    
    gbm = lgb.train(
        lgb_params,
        train_set
    )
    return gbm

def print_memory_state():
    try:
        mem = psutil.virtual_memory()
        logger.info(f"Sistema RAM: {mem.percent:.1f}% usado, {mem.available / (1024**3):.2f} GB disponibles, total {mem.total / (1024**3):.2f} GB")
    except Exception as e:
        logger.warning(f"No se pudo leer estado de la RAM: {e}")
def train_models_and_save_results(train_set,X_eval, w_eval, params, seeds, results_file, save_model, n_seeds, experiment_name, fieldnames, user):
    revs = []
    for seed in seeds[:n_seeds]:
        params["seed"] = seed
        start_time = time.time()
        logger.info(f"Entrenando modelo con seed: {seed}")
        model = train_model(train_set, params)
        y_pred = model.predict(X_eval)
        if save_model:
            joblib.dump(model, f"/home/{user}/buckets/b1/models/{experiment_name}_{seed}.pkl")
            save_model = False
        rev, _ = gan_eval(y_pred, w_eval, window=2001)
        revs.append(rev)
        if rev > 600000000:
            raise Exception(f"Ganancia excesiva: {rev}")
        if rev < 350000000:
            raise Exception(f"Ganancia insuficiente: {rev}")

        write_header = not os.path.exists(results_file)
        
        
        end_time = time.time()
        result_row = {
            "experiment_name": experiment_name,
            "seed": seed,
            "training_time": end_time - start_time,
            "moving_average_rev": rev
        }
        with open(results_file, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(result_row)
        gc.collect()
        logger.info(f"Modelo entrenado en tiempo: {end_time - start_time}")
        logger.info(f"Ganancia: {rev}")
    logger.info(f"Ganancias: {revs}")
    logger.info(f"Ganancia promedio: {np.mean(revs)}")
    logger.info(f"Ganancia máxima: {np.max(revs)}")
    logger.info(f"Ganancia mínima: {np.min(revs)}")
    logger.info(f"Ganancia std: {np.std(revs)}")
    logger.info(f"Ganancia mediana: {np.median(revs)}")
    return revs

def identify_low_importance_features(
    X_train, 
    y_train, 
    n_canaritos=10,
    params=None,
    save=False,
    output_file='low_importance_features.csv'
):
    """
    Identifica variables con menor importancia que el promedio de los canaritos.
    
    Esta función agrega canaritos (variables aleatorias) al dataset, entrena un modelo
    LightGBM, y identifica todas las variables cuya importancia es menor que el promedio
    de importancia de los canaritos.
    Parameters:
    -----------
    
    """
    
    logger.info("="*80)
    logger.info("🔍 Identificando variables de baja importancia usando canaritos")
    logger.info("="*80)
    
    # Parámetros por defecto si no se proveen
    if params is None:
        params = {
            'seed': 42,
            'min_data_in_leaf': 2000,
            'feature_fraction': 0.5,
            'canaritos': 0,  # No matar árboles por canaritos en este análisis
            'gradient_bound': 0.2
        }
        logger.info("Usando parámetros por defecto")
    
    logger.info(f"Parámetros del modelo: {params}")
    
    # Paso 1: Agregar canaritos
    logger.info(f"\n📊 Agregando {n_canaritos} canaritos al dataset...")
    canaritos_transformer = AddCanaritos(n_canaritos=n_canaritos)
    
    # Identificar columnas que no son features (para preservarlas)
    exclude_cols = ['foto_mes', 'numero_de_cliente', 'target', 'label', 'weight', 'clase_ternaria']
    feature_cols = [col for col in X_train.columns if col not in exclude_cols]
    
    logger.info(f"Features originales: {len(feature_cols)}")
    
    # Aplicar transformación de canaritos
    X_train_with_canaritos = canaritos_transformer.fit_transform(X_train)
    
    # Identificar los nombres de los canaritos
    canaritos_names = [col for col in X_train_with_canaritos.columns if col.startswith('canarito_')]
    logger.info(f"Canaritos agregados: {canaritos_names}")
    
    # Separar features para entrenar (sin columnas de identificación)
    X_train_features = X_train_with_canaritos[[col for col in X_train_with_canaritos.columns if col not in exclude_cols]]
    
    logger.info(f"Total de features para entrenar (incluyendo canaritos): {len(X_train_features.columns)}")
    
    # Paso 2: Crear dataset de LightGBM
    logger.info("\n🏋️ Creando dataset de LightGBM...")
    train_set = lgb.Dataset(
        data=X_train_features,
        label=y_train,
        free_raw_data=False
    )
    
    # Paso 3: Entrenar modelo
    logger.info("\n🚀 Entrenando modelo LightGBM...")
    start_time = time.time()
    model = train_model(train_set, params)
    training_time = time.time() - start_time
    logger.info(f"✅ Modelo entrenado en {training_time:.2f} segundos")
    logger.info(f"Número de árboles: {model.num_trees()}")
    
    # Paso 4: Obtener importancia de features
    logger.info("\n📈 Calculando importancia de features...")
    feature_names = model.feature_name()
    feature_importance = model.feature_importance(importance_type='gain')
    
    # Crear DataFrame con importancias
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })
    
    # Separar canaritos de features reales
    importance_df['is_canarito'] = importance_df['feature'].str.startswith('canarito_')
    
    # Calcular estadísticas de canaritos
    canaritos_importance = importance_df[importance_df['is_canarito']]['importance']
    canaritos_avg = canaritos_importance.mean()
    canaritos_median = canaritos_importance.median()
    canaritos_max = canaritos_importance.max()
    canaritos_min = canaritos_importance.min()
    
    logger.info(f"\n📊 Estadísticas de importancia de canaritos:")
    logger.info(f"  - Promedio: {canaritos_avg:.6f}")
    logger.info(f"  - Mediana: {canaritos_median:.6f}")
    logger.info(f"  - Máximo: {canaritos_max:.6f}")
    logger.info(f"  - Mínimo: {canaritos_min:.6f}")
    
    # Identificar features con menor importancia que el promedio de canaritos
    low_importance_mask = (importance_df['importance'] < canaritos_avg) & (~importance_df['is_canarito'])
    low_importance_features = importance_df[low_importance_mask]['feature'].tolist()
    
    logger.info(f"\n🎯 Variables con importancia menor al promedio de canaritos: {len(low_importance_features)}")
    
    # Ordenar por importancia
    importance_df_sorted = importance_df.sort_values('importance', ascending=True)
    
    # Mostrar algunas variables de baja importancia
    if len(low_importance_features) > 0:
        logger.info(f"\n📉 Top 10 variables de menor importancia:")
        low_importance_df = importance_df_sorted[importance_df_sorted['feature'].isin(low_importance_features)]
        low_importance_top10 = low_importance_df.head(10)
        for idx, row in low_importance_top10.iterrows():
            logger.info(f"  - {row['feature']}: {row['importance']:.6f}")
    else:
        logger.info("✅ Todas las variables tienen importancia mayor al promedio de canaritos")
    
    # Paso 5: Guardar resultados si se solicita
    if save:
        logger.info(f"\n💾 Guardando resultados en {output_file}...")
        
        # Preparar DataFrame para guardar
        save_df = importance_df_sorted[['feature', 'importance', 'is_canarito']].copy()
        save_df['below_canarito_avg'] = save_df['importance'] < canaritos_avg
        save_df['canaritos_avg_importance'] = canaritos_avg
        
        # Guardar
        save_df.to_csv(output_file, index=False)
        logger.info(f"✅ Resultados guardados en {output_file}")
    
    # Resumen final
    logger.info("\n" + "="*80)
    logger.info("📋 RESUMEN")
    logger.info("="*80)
    logger.info(f"Total de features originales: {len(feature_cols)}")
    logger.info(f"Número de canaritos: {n_canaritos}")
    logger.info(f"Promedio importancia canaritos: {canaritos_avg:.6f}")
    logger.info(f"Variables de baja importancia detectadas: {len(low_importance_features)}")
    logger.info(f"Porcentaje de features de baja importancia: {100 * len(low_importance_features) / len(feature_cols):.2f}%")
    logger.info("="*80)
    
    # Limpiar memoria
    gc.collect()
    
    # Retornar resultados
    return {
        'low_importance_features': low_importance_features,
        'canaritos_avg_importance': canaritos_avg,
        'feature_importance_df': importance_df_sorted[['feature', 'importance', 'is_canarito']].copy(),
        'n_low_importance': len(low_importance_features),
        'n_total_features': len(feature_cols)
    }




