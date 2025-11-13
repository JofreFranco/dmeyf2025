import os
import gc
import csv
import time
import joblib
import pandas as pd
import numpy as np
import random
from flaml import AutoML
import lightgbm as lgb
from flaml.default import preprocess_and_suggest_hyperparams
import logging
from dmeyf2025.processors.feature_processors import CleanZerosTransformer, DeltaLagTransformer, PercentileTransformer, PeriodStatsTransformer, TendencyTransformer, IntraMonthTransformer, RandomForestFeaturesTransformer, DatesTransformer, HistoricalFeaturesTransformer
from dmeyf2025.metrics.revenue import GANANCIA_ACIERTO, COSTO_ESTIMULO
"""import scipy.stats as stats
if not hasattr(stats, 'binom_test'):
    stats.binom_test = stats.binomtest  # parche compatibilidad
    np.NaN = np.nan"""
from BorutaShap import BorutaShap
from dmeyf2025.processors.sampler import SamplerProcessor
from scipy.stats import wilcoxon


logger = logging.getLogger(__name__)

# Configurar logging para que se muestre en terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Para mostrar en consola
    ]
)

debug_mode = False
sampling_rate = 0.05
logging.info("comenzando")

# Alphas para early stopping
ALPHA_1 = 0.1  # Alpha menos exigente
ALPHA_2 = 0.05  # Alpha más exigente

# %% [markdown]
# # Read data

# %%
logger.info("Leyendo dataset")
df = pd.read_csv('~/datasets/competencia_02_target.csv')
if "mprestamos_personales" in df.columns:
    df = df.drop(columns=["mprestamos_personales", "cprestamos_personales"])
weight = {"BAJA+1": 1, "BAJA+2": 1.00002, "CONTINUA": 1}
df["target"] = ((df["clase_ternaria"] == "BAJA+2") | (df["clase_ternaria"] == "BAJA+1")).astype(int)

training_months = [202008, 202009, 202010, 202011,202012, 202101, 202102, 202103, 202104]
eval_month = 202106
test_month = 202108
seeds = [537919, 923347, 173629, 419351, 287887, 1244, 24341, 1241, 4512, 6554, 62325, 6525235, 14, 4521, 474574, 74543, 32462, 12455, 5124, 55678]
if debug_mode:
    # Sample 0.5% of target=0 cases per month, keep all target=1 rows
    df_list = []
    for mes, df_mes in df[df["target"] == 0].groupby("foto_mes"):
        df_sample = df_mes.sample(frac=0.005, random_state=42)
        df_list.append(df_sample)
    df_target_0_sampled = pd.concat(df_list, axis=0)
    df_target_1 = df[df["target"] == 1]
    df = pd.concat([df_target_0_sampled, df_target_1], axis=0).reset_index(drop=True)
    seeds = [42]

# %% Extra functions

# %%
def should_run_experiment(experiment_name, results_file, min_seeds=5):
    """
    Verifica si un experimento debe ejecutarse o ya fue completado.
    
    Parameters:
    -----------
    experiment_name : str
        Nombre del experimento
    results_file : str
        Ruta al archivo CSV con resultados
    min_seeds : int
        Número mínimo de seeds que se considera "completado"
        
    Returns:
    --------
    tuple (should_run: bool, reason: str, existing_stats: dict or None)
        - should_run: True si debe ejecutarse, False si ya está completo
        - reason: Razón de la decisión
        - existing_stats: Si ya existe, dict con 'mean_gain' y 'total_time'
    """
    if not os.path.exists(results_file):
        return True, "Archivo de resultados no existe", None
    
    try:
        df_results = pd.read_csv(results_file)
        experiment_data = df_results[df_results['experiment_name'] == experiment_name]
        experiment_count = len(experiment_data)
        
        if experiment_count >= min_seeds:
            mean_moving_avg_rev = experiment_data['moving_average_rev'].mean()
            total_time = experiment_data['training_time'].sum()
            reason = f"Experimento ya completado con {experiment_count} seeds"
            stats = {
                'mean_gain': mean_moving_avg_rev,
                'total_time': total_time,
                'num_seeds': experiment_count
            }
            return False, reason, stats
        else:
            return True, f"Experimento parcial con {experiment_count} seeds, necesita más", None
            
    except Exception as e:
        logger.warning(f"Error al verificar experimento: {e}")
        return True, f"Error al leer resultados: {e}", None

# %%
def compare_experiments_wilcoxon(new_gains, baseline_gains, alpha):
    """
    Compara dos experimentos usando el test de Wilcoxon.
    
    Parameters:
    -----------
    new_gains : list
        Lista de ganancias del nuevo experimento (solo moving_average_rev)
    baseline_gains : list
        Lista de ganancias del baseline (solo moving_average_rev)
    alpha : float
        Nivel de significancia
        
    Returns:
    --------
    dict con:
        - 'p_value': float, p-value del test
        - 'is_significantly_better': bool, si el nuevo es significativamente mejor
        - 'is_significantly_worse': bool, si el nuevo es significativamente peor
        - 'new_mean': float, media del nuevo experimento
        - 'baseline_mean': float, media del baseline
    """
    new_mean = np.mean(new_gains)
    baseline_mean = np.mean(baseline_gains)
    
    # Test de Wilcoxon
    # alternative='greater' para probar si new > baseline
    # alternative='less' para probar si new < baseline
    
    try:
        # Primero probar si es mejor
        stat_better, p_value_better = wilcoxon(new_gains, baseline_gains, alternative='greater')
        # Luego probar si es peor
        stat_worse, p_value_worse = wilcoxon(new_gains, baseline_gains, alternative='less')
        
        is_significantly_better = p_value_better < alpha
        is_significantly_worse = p_value_worse < alpha
        
        # Usar el p-value correspondiente a la dirección de la diferencia
        if new_mean > baseline_mean:
            p_value = p_value_better
        else:
            p_value = p_value_worse
            
    except Exception as e:
        logger.warning(f"Error en test de Wilcoxon: {e}")
        return {
            'p_value': 1.0,
            'is_significantly_better': False,
            'is_significantly_worse': False,
            'new_mean': new_mean,
            'baseline_mean': baseline_mean
        }
    
    return {
        'p_value': p_value,
        'is_significantly_better': is_significantly_better,
        'is_significantly_worse': is_significantly_worse,
        'new_mean': new_mean,
        'baseline_mean': baseline_mean
    }
def clean_target(X_train, X_eval):
    if "target" in X_train.columns:
        X_train = X_train.drop(columns=["target"])
        logger.warning("target column found in X_train, removing it")
    if "label" in X_train.columns:
        X_train = X_train.drop(columns=["label"])
        logger.warning("label column found in X_train, removing it")
    if "clase_ternaria" in X_train.columns:
        X_train = X_train.drop(columns=["clase_ternaria"])
        logger.warning("clase_ternaria column found in X_train, removing it")
    if "weight" in X_train.columns:
        X_train = X_train.drop(columns=["weight"])
        logger.warning("weight column found in X_train, removing it")
    if "numero_de_cliente" in X_train.columns:
        X_train = X_train.drop(columns=["numero_de_cliente"])
        logger.warning("numero_de_cliente column found in X_train, removing it")

    if "target" in X_eval.columns:
        X_eval = X_eval.drop(columns=["target"])
        logger.warning("target column found in X_eval, removing it")
    if "label" in X_eval.columns:
        X_eval = X_eval.drop(columns=["label"])
        logger.warning("label column found in X_eval, removing it")
    if "clase_ternaria" in X_eval.columns:
        X_eval = X_eval.drop(columns=["clase_ternaria"])
        logger.warning("clase_ternaria column found in X_eval, removing it")
    if "weight" in X_eval.columns:
        X_eval = X_eval.drop(columns=["weight"])
        logger.warning("weight column found in X_eval, removing it")
    if "numero_de_cliente" in X_eval.columns:
        X_eval = X_eval.drop(columns=["numero_de_cliente"])
        logger.warning("numero_de_cliente column found in X_eval, removing it")
    return X_train, X_eval
# %%
def gan_eval(y_pred, weight, window=2001):
    """
    Evalúa la ganancia máxima usando una media móvil centrada con ventana de tamaño `window`.
    Retorna el mejor valor encontrado.
    """
    ganancia = np.where(weight == 1.00002, GANANCIA_ACIERTO, 0) - np.where(weight < 1.00002, COSTO_ESTIMULO, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)
    sends = np.argmax(ganancia)
    opt_sends = np.argmax(ganancia)
    if opt_sends - (window-1)/2 < 0:
        min_sends = 0
    else:
        min_sends = int(opt_sends - (window-1)/2)
    if opt_sends + (window-1)/2 > len(ganancia):
        max_sends = len(ganancia)
    else:
        max_sends = int(opt_sends + (window-1)/2)
    
    mean_ganancia = np.mean(ganancia[min_sends:max_sends])
    # Calcula la media móvil centrada con la ventana especificada
    ventana = window
    pad = ventana // 2
    ganancia_padded = np.pad(ganancia, (pad, ventana - pad - 1), mode='edge')
    # Calcula la media móvil centrada
    medias_moviles = np.convolve(ganancia_padded, np.ones(ventana)/ventana, mode='valid')


    # Obtiene el máximo de la media móvil centrada
    mejor_ganancia = np.max(medias_moviles)
    return mejor_ganancia, mean_ganancia
def gan(X_val,
    y_val,
    estimator,
    labels,
    X_train,
    y_train,
    weight_val=None,
    weight_train=None,
    *args,
):  
    y_pred = estimator.predict_proba(X_train)
    ganancia_train, g_mean_train = gan_eval(y_pred, weight_train)
    y_pred = estimator.predict(X_val)
    ganancia_val, g_mean_val = gan_eval(y_pred, weight_val)
    return -ganancia_val, {"ganancia_val": ganancia_val, "ganancia_train":ganancia_train, "g_mean_train":g_mean_train, "g_mean_val":g_mean_val}

# %%
def zero_shot_experiment(experiment_name, seeds, results_file, fieldnames, settings, X_train, y_train, w_train, X_eval, y_eval, w_eval, save_model=True, baseline_name="zero_shot_baseline", enable_early_stopping=True, is_good=None):
    # Verificar si el experimento ya fue ejecutado 5 o más veces
    if os.path.exists(results_file):
        df_results = pd.read_csv(results_file)
        experiment_count = len(df_results[df_results['experiment_name'] == experiment_name])
        if experiment_count >= 5:
            logging.info(f"Experimento {experiment_name} ya fue ejecutado {experiment_count} veces. Salteando...")
            # Retornar los promedios existentes
            experiment_data = df_results[df_results['experiment_name'] == experiment_name]
            mean_moving_avg_rev = experiment_data['moving_average_rev'].mean()
            total_time = experiment_data['training_time'].sum()
            return mean_moving_avg_rev, total_time
    
    automl = AutoML()
    # Entrenamiento
    logging.info(f"Iniciando experimento {experiment_name}...")
    gains = []
    times = []
    
    # Obtener ganancias del baseline para comparación con early stopping
    baseline_gains = None
    if enable_early_stopping and experiment_name != baseline_name and os.path.exists(results_file):
        df_results = pd.read_csv(results_file)
        baseline_data = df_results[df_results['experiment_name'] == baseline_name]
        if len(baseline_data) >= 5:
            baseline_gains = baseline_data['moving_average_rev'].tolist()
            logging.info(f"Baseline '{baseline_name}' encontrado con {len(baseline_gains)} seeds")
        else:
            logging.warning(f"Baseline '{baseline_name}' no tiene suficientes seeds para early stopping. Deshabilitando early stopping.")
            enable_early_stopping = False
    
    X_train, X_eval = clean_target(X_train, X_eval)
    # Implementación del bucle con early stopping
    seed_index = 0
    min_seeds = 5  # Mínimo de seeds a ejecutar antes de evaluar
    seeds_per_batch = 2  # Seeds a ejecutar en cada batch después del mínimo
    should_continue = True
    early_stop_reason = None
    
    while should_continue and seed_index < len(seeds):
        # Determinar cuántas seeds ejecutar en este batch
        if seed_index < min_seeds:
            # Primer batch: ejecutar min_seeds
            batch_size = min_seeds - seed_index
        else:
            # Batches subsiguientes: ejecutar seeds_per_batch
            batch_size = min(seeds_per_batch, len(seeds) - seed_index)
        
        # Ejecutar el batch de seeds
        for i in range(batch_size):
            if seed_index >= len(seeds):
                break
                
            seed = seeds[seed_index]
            seed_index += 1
            
            training_start_time = time.time()
            settings["seed"] = seed
            (
            hyperparams,
            estimator_class,
            X_transformed,
            y_transformed,
            feature_transformer,
            label_transformer,
            ) = preprocess_and_suggest_hyperparams("classification", X_train, y_train, "lgbm")
            hyperparams["random_state"] = seed
            model = estimator_class(**hyperparams)  # estimator_class is lightgbm.LGBMClassifier

            model.fit(X_transformed, y_train)  # LGBMClassifier can handle raw labels
            X_val = feature_transformer.transform(X_eval)  # preprocess test data
            y_pred = model.predict_proba(X_val)[:,1]
            rev = gan_eval(y_pred, w_eval, window=2001)
            training_end_time = time.time()
            training_time = training_end_time - training_start_time
            print(f"Seed: {seed}")
            print("Ganancia:", rev, "Tiempo de entrenamiento:", training_time)
            gains.append(rev)
            times.append(training_time)
            # Prepare row data
            result_row = {
                "experiment_name": experiment_name,
                "seed": seed,
                "training_time": training_time,
                "hyperparameters": repr(model.get_params()),
                "moving_average_rev": float(rev[0]),
                "mean_over_best_gain": float(rev[1])
            }

            write_header = not os.path.exists(results_file)
            with open(results_file, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(result_row)
            if save_model:
                joblib.dump(model, f"/home/martin232009/buckets/b1/models/{experiment_name}_{seed}.pkl")
                save_model = False
            gc.collect()
        
        # Evaluar con Wilcoxon después de cada batch (si early stopping está habilitado)
        if enable_early_stopping and baseline_gains is not None and seed_index >= min_seeds:
            # Extraer solo los moving_average_rev de gains
            current_gains = [g[0] for g in gains]
            
            # Ajustar baseline_gains al mismo tamaño que current_gains para la comparación
 
            baseline_gains_truncated = baseline_gains[:len(current_gains)]
            
            # Comparar con alpha menos exigente (ALPHA_1)
            comparison_alpha1 = compare_experiments_wilcoxon(current_gains, baseline_gains_truncated, ALPHA_1)
            
            logging.info(f"\n{'='*80}")
            logging.info(f"Evaluación Early Stopping - Seeds ejecutadas: {seed_index}")
            logging.info(f"{'='*80}")
            logging.info(f"Media nuevo experimento: {comparison_alpha1['new_mean']:,.0f}")
            logging.info(f"Media baseline: {comparison_alpha1['baseline_mean']:,.0f}")
            logging.info(f"P-value (alpha={ALPHA_1}): {comparison_alpha1['p_value']:.4f}")
            
            # Caso 1: ¿Es significativamente peor que el baseline?
            if comparison_alpha1['is_significantly_worse']:
                should_continue = False
                early_stop_reason = f"Experimento significativamente PEOR que baseline (p={comparison_alpha1['p_value']:.4f} < {ALPHA_1})"
                logging.info(f"⛔ EARLY STOP: {early_stop_reason}")
                if is_good is not None:
                    is_good.append(False)
            
            # Caso 2: No es significativamente peor
            else:
                # ¿La media es mejor que el baseline?
                if comparison_alpha1['new_mean'] > comparison_alpha1['baseline_mean']:
                    # Comparar con alpha más exigente (ALPHA_2)
                    comparison_alpha2 = compare_experiments_wilcoxon(current_gains, baseline_gains_truncated, ALPHA_2)
                    logging.info(f"Media mejor que baseline. Probando con alpha más exigente ({ALPHA_2})...")
                    logging.info(f"P-value (alpha={ALPHA_2}): {comparison_alpha2['p_value']:.4f}")
                    
                    if comparison_alpha2['is_significantly_better']:
                        should_continue = False
                        early_stop_reason = f"Experimento significativamente MEJOR que baseline con alpha exigente (p={comparison_alpha2['p_value']:.4f} < {ALPHA_2})"
                        logging.info(f"✅ EARLY STOP: {early_stop_reason}")
                        if is_good is not None:
                            is_good.append(True)
                    else:
                        logging.info(f"↗️ Experimento promete pero no es significativo con alpha={ALPHA_2}. Continuando con {seeds_per_batch} seeds más...")
                else:
                    logging.info(f"↘️ Media peor que baseline pero no significativamente. Continuando con {seeds_per_batch} seeds más...")
            
            # Si alcanzamos el límite de seeds disponibles
            if seed_index >= len(seeds) and should_continue:
                should_continue = False
                early_stop_reason = f"Se agotaron todas las seeds disponibles ({len(seeds)})"
                logging.info(f"🏁 Finalizando: {early_stop_reason}")
                if is_good is not None:
                    is_good.append(True)
        # Si no hay early stopping habilitado, ejecutar todas las seeds
        elif not enable_early_stopping and seed_index >= len(seeds):
            should_continue = False
    
    # Log final
    if early_stop_reason:
        logging.info(f"\n{'='*80}")
        logging.info(f"Experimento finalizado: {early_stop_reason}")
        logging.info(f"Total seeds ejecutadas: {seed_index} de {len(seeds)}")
        logging.info(f"{'='*80}\n")
    # gains es una lista de tuplas (moving_average_rev, mean_over_best_gain)
    # Retornamos el promedio del moving_average_rev
    mean_moving_avg_rev = np.mean([g[0] for g in gains])
    return mean_moving_avg_rev, np.sum(times), is_good

# %%

def prepare_data(df, training_months, eval_month, test_month, get_features):
    df = df.copy()
    df["label"] = ((df["clase_ternaria"] == "BAJA+2") | (df["clase_ternaria"] == "BAJA+1")).astype(int)
    df["weight"] = np.array([weight[item] for item in df["clase_ternaria"]])
    df = df.drop(columns=["clase_ternaria"])
    df_transformed = get_features(df, training_months)
    df_train = df_transformed[df_transformed["foto_mes"].isin(training_months)]
    df_eval = df_transformed[df_transformed["foto_mes"] == eval_month]
    df_test = df_transformed[df_transformed["foto_mes"] == test_month]

    y_eval = df_eval["label"]
    w_eval = df_eval["weight"]
    X_eval = df_eval.drop(columns=["label", "weight"])


    y_test = df_test["label"]
    w_test = df_test["weight"]
    X_test = df_test.drop(columns=["label", "weight"])

    y_train = df_train["label"]
    X_train = df_train.drop(columns=["label"])
    X_train, y_train = SamplerProcessor(sampling_rate).fit_transform(X_train, y_train)
    w_train = X_train["weight"]
    X_train = X_train.drop(columns=["weight"])
    return X_train, y_train, w_train, X_eval, y_eval, w_eval, X_test, y_test

# %% Settings

settings = {
    "time_budget": None,               # segundos
    "max_iter": 0,
    "starting_points": "data",        # Arrancamos con zero-shot
    "metric": gan,                    # métrica custom
    "task": "classification",         # binaria
    "estimator_list": ["lgbm"],
    "log_file_name": "zero-shot.log",
    "eval_method": "holdout",         
    "verbose": 1,
    "n_jobs": -1,
}

results_file = "/home/martin232009/buckets/b1/results.csv"
fieldnames = ["experiment_name", "seed", "training_time", "hyperparameters", "moving_average_rev", "mean_over_best_gain"]

# Inicializar lista is_good vacía para ir pasando entre experimentos
is_good = []

# %% Baseline
# # Baseline
# 
# - Sacar prestamos personales
# - Lags y Delta Lags de orden 2

# %%
experiment_name = "zero_shot_baseline"
try:
    # Verificar si el experimento ya fue ejecutado
    should_run, reason, existing_stats = should_run_experiment(experiment_name, results_file, min_seeds=20)
    if not should_run:
        logging.info(f"⏭️  Salteando experimento '{experiment_name}': {reason}")
        logging.info(f"   Ganancia promedio: {existing_stats['mean_gain']:,.0f}, Tiempo total: {existing_stats['total_time']:.2f}s")
        mean_rev = existing_stats['mean_gain']
        total_time = existing_stats['total_time']
    else:
        logging.info(f"▶️  Ejecutando experimento '{experiment_name}': {reason}")
        
        def get_features(X, training_months):
            logger.info("Iniciando delta lag transformer...")
            delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2, exclude_cols= ["foto_mes", "numero_de_cliente", "target", "label", "weight", "clase_ternaria"])
            X_transformed = delta_lag_transformer.fit_transform(X)
            logger.info(f"Cantidad de features después de delta lag transformer: {len(X_transformed.columns)}")
            return X_transformed

        # %%
        X_train, y_train, w_train, X_eval, y_eval, w_eval, X_test, y_test = prepare_data(df, training_months, eval_month, test_month, get_features)

        # %%
        # Baseline: no usar early stopping ya que es el baseline de comparación
        mean_rev, total_time, _ = zero_shot_experiment(experiment_name, seeds, results_file, fieldnames, settings, X_train, y_train, w_train, X_eval, y_eval, w_eval, enable_early_stopping=False)
        print(f"Ganancia promedio: {mean_rev}, Tiempo total: {total_time}")
except Exception as e:
    logging.error(f"❌ ERROR en experimento '{experiment_name}': {str(e)}")
    logging.error(f"   Traceback: ", exc_info=True)
    logging.info(f"   Continuando con el siguiente experimento...\n")
    mean_rev = None
    total_time = None

# %% Zero-Clean
# # Zero-Clean
# 
# - Sacar prestamos personales
# - Pasar ceros a Nan en los casos que corresponda
# - Lags y Delta Lags de orden 2
# 

# %%
try:
    experiment_name = "zero_shot_zero_clean"

    # Verificar si el experimento ya fue ejecutado
    should_run, reason, existing_stats = should_run_experiment(experiment_name, results_file, min_seeds=5)
    if not should_run:
        logging.info(f"⏭️  Salteando experimento '{experiment_name}': {reason}")
        logging.info(f"   Ganancia promedio: {existing_stats['mean_gain']:,.0f}, Tiempo total: {existing_stats['total_time']:.2f}s")
        mean_rev = existing_stats['mean_gain']
        total_time = existing_stats['total_time']
    else:
        logging.info(f"▶️  Ejecutando experimento '{experiment_name}': {reason}")
        
        def get_features(X, training_months):
            clean_zeros_transformer = CleanZerosTransformer()
            X_transformed = clean_zeros_transformer.fit_transform(X)
            logger.info("Iniciando delta lag transformer...")
            delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2, exclude_cols= ["foto_mes", "numero_de_cliente", "target", "label", "weight"])
            X_transformed = delta_lag_transformer.fit_transform(X_transformed)
            logger.info(f"Cantidad de features después de delta lag transformer: {len(X_transformed.columns)}")
            return X_transformed

        # %%
        X_train, y_train, w_train, X_eval, y_eval, w_eval, X_test, y_test = prepare_data(df, training_months, eval_month, test_month, get_features)

        # %%
        mean_rev, total_time, _ = zero_shot_experiment(experiment_name, seeds, results_file, fieldnames, settings, X_train, y_train, w_train, X_eval, y_eval, w_eval, enable_early_stopping=True)
        print(f"Ganancia promedio: {mean_rev}, Tiempo total: {total_time}")
except Exception as e:
    logging.error(f"❌ ERROR en experimento '{experiment_name}': {str(e)}")
    logging.info(f"   Continuando con el siguiente experimento...\n")
    mean_rev = None
    total_time = None

# %% Percentiles None
# # Percentiles None
# 
# - Sacar prestamos personales
# - Pasar ceros a Nan en los casos que corresponda
# - Lags y Delta Lags de orden 2
# - Percentiles discretizados en saltos de None
# 
# 

# %%
try:
    experiment_name = "zero_shot_percentiles_None_isolated"

    # Verificar si el experimento ya fue ejecutado
    should_run, reason, existing_stats = should_run_experiment(experiment_name, results_file, min_seeds=5)
    if not should_run:
        logging.info(f"⏭️  Salteando experimento '{experiment_name}': {reason}")
        logging.info(f"   Ganancia promedio: {existing_stats['mean_gain']:,.0f}, Tiempo total: {existing_stats['total_time']:.2f}s")
        mean_rev = existing_stats['mean_gain']
        total_time = existing_stats['total_time']
    else:
        logging.info(f"▶️  Ejecutando experimento '{experiment_name}': {reason}")
        
        def get_features(X, training_months):
            logger.info("Iniciando delta lag transformer...")
            delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2, exclude_cols= ["foto_mes", "numero_de_cliente", "target", "label", "weight"])
            X_transformed = delta_lag_transformer.fit_transform(X)
            logger.info(f"Cantidad de features después de delta lag transformer: {len(X_transformed.columns)}")

            # Percentiles discretizados en saltos de None
            percentiles_transformer = PercentileTransformer(n_bins=None, replace_original=True)
            X_transformed = percentiles_transformer.fit_transform(X_transformed)
            logger.info(f"Cantidad de features después de percentiles transformer: {len(X_transformed.columns)}")
            return X_transformed

        # %%
        X_train, y_train, w_train, X_eval, y_eval, w_eval, X_test, y_test = prepare_data(df, training_months, eval_month, test_month, get_features)

        # %%
        mean_rev, total_time, _ = zero_shot_experiment(experiment_name, seeds, results_file, fieldnames, settings, X_train, y_train, w_train, X_eval, y_eval, w_eval, enable_early_stopping=True)
        print(f"Ganancia promedio: {mean_rev}, Tiempo total: {total_time}")

    ganancia_intra_month_None = mean_rev
except Exception as e:
    logging.error(f"❌ ERROR en experimento '{experiment_name}': {str(e)}")

    logging.info(f"   Continuando con el siguiente experimento...\n")
    mean_rev = None
    total_time = None
# %% Intra Month F.E
# # Intra Month F.E
# 
# - Sacar prestamos personales
# - Pasar ceros a Nan en los casos que corresponda
# - Feature engineering intra mes
# - Lags y Delta Lags de orden 2
# - Percentiles discretizados en saltos de 1 o 5, el que de mejores resultados
# 
# 
# 

# %%
try:
    experiment_name = "zero_shot_intra_month-p-zc"

    # Verificar si el experimento ya fue ejecutado
    should_run, reason, existing_stats = should_run_experiment(experiment_name, results_file, min_seeds=5)
    if not should_run:
        logging.info(f"⏭️  Salteando experimento '{experiment_name}': {reason}")
        logging.info(f"   Ganancia promedio: {existing_stats['mean_gain']:,.0f}, Tiempo total: {existing_stats['total_time']:.2f}s")
        mean_rev = existing_stats['mean_gain']
        total_time = existing_stats['total_time']
    else:
        logging.info(f"▶️  Ejecutando experimento '{experiment_name}': {reason}")
        
        n_bins = None

        def get_features(X, training_months):
            clean_zeros_transformer = CleanZerosTransformer()
            X_transformed = clean_zeros_transformer.fit_transform(X)
            
            logger.info("Iniciando delta lag transformer...")
            intra_month_transformer = IntraMonthTransformer()
            X_transformed = intra_month_transformer.fit_transform(X)
            logger.info(f"Cantidad de features después de intra month transformer: {len(X_transformed.columns)}")
            logger.info("Iniciando dates transformer...")
            dates_transformer = DatesTransformer(exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"])
            X_transformed = dates_transformer.fit_transform(X_transformed)
            logger.info(f"Cantidad de features después de dates transformer: {len(X_transformed.columns)}")
            logger.info("Iniciando delta lag transformer...")
            delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2, exclude_cols= ["foto_mes", "numero_de_cliente", "target", "label", "weight"])
            X_transformed = delta_lag_transformer.fit_transform(X_transformed)
            logger.info(f"Cantidad de features después de delta lag transformer: {len(X_transformed.columns)}")
            # Percentiles discretizados en saltos de None
            percentiles_transformer = PercentileTransformer(n_bins=None, replace_original=True)
            X_transformed = percentiles_transformer.fit_transform(X_transformed)
            logger.info(f"Cantidad de features después de percentiles transformer: {len(X_transformed.columns)}")
            return X_transformed

        # %%
        X_train, y_train, w_train, X_eval, y_eval, w_eval, X_test, y_test = prepare_data(df, training_months, eval_month, test_month, get_features)

        # %%
        mean_rev, total_time, is_good= zero_shot_experiment(experiment_name, seeds, results_file, fieldnames, settings, X_train, y_train, w_train, X_eval, y_eval, w_eval, enable_early_stopping=True, is_good=is_good)
        print(f"Ganancia promedio: {mean_rev}, Tiempo total: {total_time}")
        logging.info(f"is_good después de IntraMonth: {is_good}")
except Exception as e:
    logging.error(f"❌ ERROR en experimento '{experiment_name}': {str(e)}")

    logging.info(f"   Continuando con el siguiente experimento...\n")
    mean_rev = None
    total_time = None
    is_good.append(False)
# %% Historical
# # Historical
# 
# - Sacar prestamos personales
# - Pasar ceros a Nan en los casos que corresponda
# - Feature engineering intra mes
# - Lags y Delta Lags de orden 2
# - Tendencias
# - Stats de periodos
# - Percentiles discretizados en saltos de 1 o 5, el que de mejores resultados
# 
# 
# 

# %%
try:
        
    experiment_name = "zero_shot_historical_v2"

    # Verificar si el experimento ya fue ejecutado
    should_run, reason, existing_stats = should_run_experiment(experiment_name, results_file, min_seeds=5)
    if not should_run:
        logging.info(f"⏭️  Salteando experimento '{experiment_name}': {reason}")
        logging.info(f"   Ganancia promedio: {existing_stats['mean_gain']:,.0f}, Tiempo total: {existing_stats['total_time']:.2f}s")
        mean_rev = existing_stats['mean_gain']
        total_time = existing_stats['total_time']
    else:
        logging.info(f"▶️  Ejecutando experimento '{experiment_name}': {reason}")
        
        n_bins = None

        def get_features(X, training_months):
            clean_zeros_transformer = CleanZerosTransformer()
            X_transformed = clean_zeros_transformer.fit_transform(X)
            initial_columns = X.columns
            logger.info("Iniciando delta lag transformer...")
            delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2, exclude_cols= ["foto_mes", "numero_de_cliente", "target", "label", "weight"])

            X_transformed = delta_lag_transformer.fit_transform(X)
            logger.info(f"Cantidad de features después de delta lag transformer: {len(X_transformed.columns)}")
            logger.info("Iniciando tendency transformer...")
            new_columns = set(X_transformed.columns) - set(initial_columns)
            tendency_transformer = TendencyTransformer(exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"] + list(new_columns))
            X_transformed = tendency_transformer.fit_transform(X_transformed)
            new_columns = set(X_transformed.columns) - set(initial_columns)

            logger.info(f"Cantidad de features después de tendency transformer: {len(X_transformed.columns)}")

            logger.info("Iniciando period stats transformer...")
            period_stats_transformer = PeriodStatsTransformer(periods=[2, 3], exclude_cols=list(new_columns) + ["foto_mes", "numero_de_cliente", "target", "label", "weight"])
            X_transformed = period_stats_transformer.fit_transform(X_transformed)
            new_columns = set(X_transformed.columns) - set(initial_columns)
            logger.info(f"Cantidad de features después de period stats transformer: {len(X_transformed.columns)}")
            logger.info("Iniciando historical features transformer...")
            historical_features_transformer = HistoricalFeaturesTransformer(exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"])
            X_transformed = historical_features_transformer.fit_transform(X_transformed)
            logger.info(f"Cantidad de features después de historical features transformer: {len(X_transformed.columns)}")
            percentiles_transformer = PercentileTransformer(n_bins=None, replace_original=True)
            X_transformed = percentiles_transformer.fit_transform(X_transformed)
            logger.info(f"Cantidad de features después de percentiles transformer: {len(X_transformed.columns)}")
            return X_transformed

        # %%
        X_train, y_train, w_train, X_eval, y_eval, w_eval, X_test, y_test = prepare_data(df, training_months, eval_month, test_month, get_features)

        # %%
        mean_rev, total_time, is_good= zero_shot_experiment(experiment_name, seeds, results_file, fieldnames, settings, X_train, y_train, w_train, X_eval, y_eval, w_eval, enable_early_stopping=True, is_good=is_good)
        print(f"Ganancia promedio: {mean_rev}, Tiempo total: {total_time}")
        logging.info(f"is_good después de Historical: {is_good}")
except Exception as e:
    logging.error(f"❌ ERROR en experimento '{experiment_name}': {str(e)}")

    logging.info(f"   Continuando con el siguiente experimento...\n")
    mean_rev = None
    total_time = None
    is_good.append(False)
# %% Dates
# # Dates
# 
# - Sacar prestamos personales
# - Pasar ceros a Nan en los casos que corresponda
# - Dates
# 
# 

# %%
try:
    lasfh
        
    experiment_name = "zero_shot_dates"

    # Verificar si el experimento ya fue ejecutado
    should_run, reason, existing_stats = should_run_experiment(experiment_name, results_file, min_seeds=5)
    if not should_run:
        logging.info(f"⏭️  Salteando experimento '{experiment_name}': {reason}")
        logging.info(f"   Ganancia promedio: {existing_stats['mean_gain']:,.0f}, Tiempo total: {existing_stats['total_time']:.2f}s")
        mean_rev = existing_stats['mean_gain']
        total_time = existing_stats['total_time']
    else:
        logging.info(f"▶️  Ejecutando experimento '{experiment_name}': {reason}")
        
        def get_features(X, training_months):

            logger.info("Iniciando delta lag transformer...")
            delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2, exclude_cols= ["foto_mes", "numero_de_cliente", "target", "label", "weight"])
        
            X_transformed = delta_lag_transformer.fit_transform(X)
            logger.info(f"Cantidad de features después de delta lag transformer: {len(X_transformed.columns)}")
            logger.info("Iniciando dates transformer...")
            dates_transformer = DatesTransformer(exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"])
            X_transformed = dates_transformer.fit_transform(X_transformed)
            logger.info(f"Cantidad de features después de dates transformer: {len(X_transformed.columns)}")

            return X_transformed

        # %%
        X_train, y_train, w_train, X_eval, y_eval, w_eval, X_test, y_test = prepare_data(df, training_months, eval_month, test_month, get_features)

        # %%
        mean_rev, total_time, is_good= zero_shot_experiment(experiment_name, seeds, results_file, fieldnames, settings, X_train, y_train, w_train, X_eval, y_eval, w_eval, enable_early_stopping=True, is_good=is_good)
        print(f"Ganancia promedio: {mean_rev}, Tiempo total: {total_time}")
        logging.info(f"is_good después de Dates: {is_good}")
except Exception as e:
    logging.error(f"❌ ERROR en experimento '{experiment_name}': {str(e)}")

    logging.info(f"   Continuando con el siguiente experimento...\n")
    mean_rev = None
    total_time = None
    is_good.append(False)

# %% Other Features
# # Other Features
# 
# - Sacar prestamos personales
# - Pasar ceros a Nan en los casos que corresponda
# - Other Features
# 
# 

# %%
try:
    print(asd)
    experiment_name = "zero_shot_other_features"

    # Verificar si el experimento ya fue ejecutado
    should_run, reason, existing_stats = should_run_experiment(experiment_name, results_file, min_seeds=5)
    if not should_run:
        logging.info(f"⏭️  Salteando experimento '{experiment_name}': {reason}")
        logging.info(f"   Ganancia promedio: {existing_stats['mean_gain']:,.0f}, Tiempo total: {existing_stats['total_time']:.2f}s")
        mean_rev = existing_stats['mean_gain']
        total_time = existing_stats['total_time']
    else:
        logging.info(f"▶️  Ejecutando experimento '{experiment_name}': {reason}")
        
        def get_features(X, training_months):
            clean_zeros_transformer = CleanZerosTransformer()
            X_transformed = clean_zeros_transformer.fit_transform(X)
            other_features_transformer = HistoricalFeaturesTransformer(exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"])
            X_transformed = other_features_transformer.fit_transform(X)
            logger.info(f"Cantidad de features después de other features transformer: {len(X_transformed.columns)}")
            logger.info("Iniciando delta lag transformer...")
            delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2, exclude_cols= ["foto_mes", "numero_de_cliente", "target", "label", "weight"])
        
            X_transformed = delta_lag_transformer.fit_transform(X_transformed)
            logger.info(f"Cantidad de features después de delta lag transformer: {len(X_transformed.columns)}")
            percentiles_transformer = PercentileTransformer(n_bins=None, replace_original=True)
            X_transformed = percentiles_transformer.fit_transform(X_transformed)
            logger.info(f"Cantidad de features después de percentiles transformer: {len(X_transformed.columns)}")
            return X_transformed

        # %%
        X_train, y_train, w_train, X_eval, y_eval, w_eval, X_test, y_test = prepare_data(df, training_months, eval_month, test_month, get_features)

        # %%
        mean_rev, total_time, is_good= zero_shot_experiment(experiment_name, seeds, results_file, fieldnames, settings, X_train, y_train, w_train, X_eval, y_eval, w_eval, enable_early_stopping=True, is_good=is_good)
        print(f"Ganancia promedio: {mean_rev}, Tiempo total: {total_time}")
        logging.info(f"is_good después de Dates: {is_good}")
except Exception as e:
    logging.error(f"❌ ERROR en experimento '{experiment_name}': {str(e)}")

    logging.info(f"   Continuando con el siguiente experimento...\n")
    mean_rev = None
    total_time = None
    is_good.append(False)
# %% Random Forest Features
# # Random Forest Features
# 
# - Sacar prestamos personales
# - Pasar ceros a Nan en los casos que corresponda
# - Feature engineering intra mes
# - Lags y Delta Lags de orden 2
# - Tendencias
# - Stats de periodos
# - Percentiles discretizados en saltos de 1 o 5, el que de mejores resultados
# - Random Forest Features
# 
# 
# 

# %%
try:
    experiment_name = "zero_shot_random_forest_features"

    # Verificar si el experimento ya fue ejecutado
    should_run, reason, existing_stats = should_run_experiment(experiment_name, results_file, min_seeds=5)
    if not should_run:
        logging.info(f"⏭️  Salteando experimento '{experiment_name}': {reason}")
        logging.info(f"   Ganancia promedio: {existing_stats['mean_gain']:,.0f}, Tiempo total: {existing_stats['total_time']:.2f}s")
        mean_rev = existing_stats['mean_gain']
        total_time = existing_stats['total_time']
    else:
        logging.info(f"▶️  Ejecutando experimento '{experiment_name}': {reason}")
        
        n_bins = None

        def get_features(X, training_months):
            clean_zeros_transformer = CleanZerosTransformer()
            X_transformed = clean_zeros_transformer.fit_transform(X)
            delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2, exclude_cols= ["foto_mes", "numero_de_cliente", "target", "label", "weight"])

            X_transformed = delta_lag_transformer.fit_transform(X)
            logger.info(f"Cantidad de features después de delta lag transformer: {len(X_transformed.columns)}")
            logger.info("Iniciando tendency transformer...")
    
            
            percentiles_transformer = PercentileTransformer(n_bins=None, replace_original=True)
            X_transformed = percentiles_transformer.fit_transform(X_transformed)
            logger.info(f"Cantidad de features después de percentiles transformer: {len(X_transformed.columns)}")
            logger.info("Iniciando RandomForest Feature Transformer...")
            random_forest_features_transformer = RandomForestFeaturesTransformer(training_months= training_months)  
            X_transformed = random_forest_features_transformer.fit_transform(X_transformed)
            logger.info(f"Cantidad de features después de RandomForest Feature Transformer: {len(X_transformed.columns)}")
            return X_transformed

        # %%
        X_train, y_train, w_train, X_eval, y_eval, w_eval, X_test, y_test = prepare_data(df, training_months, eval_month, test_month, get_features)

        # %%
        
        mean_rev, total_time, is_good= zero_shot_experiment(experiment_name, seeds, results_file, fieldnames, settings, X_train, y_train, w_train, X_eval, y_eval, w_eval, enable_early_stopping=True, is_good=is_good)
        print(f"Ganancia promedio: {mean_rev}, Tiempo total: {total_time}")
        logging.info(f"is_good después de RandomForest: {is_good}")

except Exception as e:
    logging.error(f"❌ ERROR en experimento '{experiment_name}': {str(e)}")
    logging.info(f"   Continuando con el siguiente experimento...\n")
    mean_rev = None
    total_time = None
    is_good.append(False)

# %% All Features
# # All Features
# 
# - Sacar prestamos personales
# - Pasar ceros a Nan en los casos que corresponda
# - Feature engineering intra mes
# - Lags y Delta Lags de orden 2
# - Tendencias
# - Stats de periodos
# - Percentiles discretizados en saltos de 1 o 5, el que de mejores resultados
# - Random Forest Features
# 
try:
    experiment_name = "zero_shot_all"
    
    # Verificar si tenemos suficientes elementos en is_good
    logging.info(f"Estado de is_good antes de Combined Winners: {is_good}")
    
    if False:
        logging.warning(f"⚠️  No hay suficientes experimentos completados (is_good tiene {len(is_good)} elementos, se necesitan 3)")
        logging.info(f"⏭️  Salteando experimento '{experiment_name}'")
    else:
        # Verificar si el experimento ya fue ejecutado
        should_run, reason, existing_stats = should_run_experiment(experiment_name, results_file, min_seeds=5)
        
        # Verificar si al menos uno de los experimentos fue exitoso
        
        
        if not should_run:
            logging.info(f"⏭️  Salteando experimento '{experiment_name}': {reason}")
            logging.info(f"   Ganancia promedio: {existing_stats['mean_gain']:,.0f}, Tiempo total: {existing_stats['total_time']:.2f}s")
            mean_rev = existing_stats['mean_gain']
            total_time = existing_stats['total_time']
    
        else:
            logging.info(f"▶️  Ejecutando experimento '{experiment_name}': {reason}")
            
            
            # Lista de booleanos para controlar qué procesamiento aplicar
            # Orden: [IntraMonth, Historical, RandomForest]
            # DeltaLag siempre se aplica
            
            def get_features(X, training_months):
                X_transformed = X.copy()
                initial_columns = set(X.columns)
                clean_zeros_transformer = CleanZerosTransformer()
                X_transformed = clean_zeros_transformer.fit_transform(X)
                # 1. IntraMonth (si is_good[0])
                
                logger.info("Aplicando IntraMonth...")
                intra_month_transformer = IntraMonthTransformer()
                X_transformed = intra_month_transformer.fit_transform(X_transformed)
                logger.info(f"Cantidad de features después de IntraMonth: {len(X_transformed.columns)}")
                
                # 1. Other Features (si is_good[0])
                logger.info("Aplicando Other Features...")
                other_features_transformer = HistoricalFeaturesTransformer(exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"])
                X_transformed = other_features_transformer.fit_transform(X_transformed)
                logger.info(f"Cantidad de features después de Other Features: {len(X_transformed.columns)}")
                # 2. Dates (si is_good[1])
                logger.info("Aplicando Dates...")
                dates_transformer = DatesTransformer(exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"])
                X_transformed = dates_transformer.fit_transform(X_transformed)
                logger.info(f"Cantidad de features después de Dates: {len(X_transformed.columns)}")
                # 2. DeltaLag (SIEMPRE se aplica)
                logger.info("Aplicando DeltaLag...")
                delta_lag_transformer = DeltaLagTransformer(n_deltas=2, n_lags=2, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"])
                X_transformed = delta_lag_transformer.fit_transform(X_transformed)
                logger.info(f"Cantidad de features después de DeltaLag: {len(X_transformed.columns)}")
                new_columns_after_deltalag = set(X_transformed.columns) - initial_columns
                
                # 3. Historical: Tendency + PeriodStats (si is_good[1])

                logger.info("Aplicando Tendency...")
                tendency_transformer = TendencyTransformer(exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"] + list(new_columns_after_deltalag))
                X_transformed = tendency_transformer.fit_transform(X_transformed)
                logger.info(f"Cantidad de features después de Tendency: {len(X_transformed.columns)}")
                """
                logger.info("Aplicando PeriodStats...")
                new_columns = set(X_transformed.columns) - initial_columns
                period_stats_transformer = PeriodStatsTransformer(periods=[2, 3], exclude_cols=list(new_columns) + ["foto_mes", "numero_de_cliente", "target", "label", "weight"])
                X_transformed = period_stats_transformer.fit_transform(X_transformed)
                logger.info(f"Cantidad de features después de PeriodStats: {len(X_transformed.columns)}")
                                """
                # 4. RandomForest Features (si is_good[2])
                logger.info("Aplicando Percentiles...")
                percentiles_transformer = PercentileTransformer(n_bins=None, replace_original=True)
                X_transformed = percentiles_transformer.fit_transform(X_transformed)
                logger.info(f"Cantidad de features después de Percentiles: {len(X_transformed.columns)}")

                logger.info("Aplicando RandomForest Features...")
                random_forest_features_transformer = RandomForestFeaturesTransformer(training_months=training_months)
                X_transformed = random_forest_features_transformer.fit_transform(X_transformed)
                logger.info(f"Cantidad de features después de RandomForest: {len(X_transformed.columns)}")
                
                
                return X_transformed
            
            # Preparar datos
            X_train, y_train, w_train, X_eval, y_eval, w_eval, X_test, y_test = prepare_data(df, training_months, eval_month, test_month, get_features)
            
            # Ejecutar experimento (no pasar is_good porque este experimento no debe modificarlo)
            result = zero_shot_experiment(experiment_name, seeds, results_file, fieldnames, settings, X_train, y_train, w_train, X_eval, y_eval, w_eval, enable_early_stopping=True)
            mean_rev, total_time = result[0], result[1]
            print(f"Ganancia promedio: {mean_rev}, Tiempo total: {total_time}")
        
except Exception as e:
    logging.error(f"❌ ERROR en experimento '{experiment_name}': {str(e)}")
    logging.info(f"   Continuando con el siguiente experimento...\n")
    mean_rev = None
    total_time = None