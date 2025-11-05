import logging
import ast
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)


def parse_list_string(list_str):
    """
    Parsea un string que contiene una lista de valores numpy o listas de Python.
    
    Parameters:
    -----------
    list_str : str
        String que representa una lista (ej: "[260560000.0, 260960000.0]")
        
    Returns:
    --------
    list
        Lista de valores numéricos
    """
    try:
        # Primero intentar ast.literal_eval
        parsed = ast.literal_eval(list_str)
        # Convertir a lista numpy y luego a lista de Python
        result = list(np.array(parsed))
        return result
    except (ValueError, SyntaxError):
        # Si falla, intentar remover prefijos de numpy
        cleaned = list_str.replace('np.int64', '').replace('np.float64', '').strip()
        parsed = ast.literal_eval(cleaned)
        result = list(np.array(parsed))
        return result


def analyze_experiments(tracking_file=None, alpha=0.05):
    """
    Lee el archivo de trackeo de experimentos, ordena por ganancia promedio,
    aplica test de Wilcoxon entre el primero y el resto, y loggea resultados.
    
    Parameters:
    -----------
    tracking_file : str, optional
        Ruta al archivo CSV con los resultados de los experimentos.
        Si no se proporciona, usa 'results/experiments_tracking.csv'
    alpha : float, default=0.05
        Nivel de significancia para el test de Wilcoxon
        
    Returns:
    --------
    None
    """
    if tracking_file is None:
        tracking_file = 'results/experiments_tracking.csv'
    # Leer el archivo
    logger.info(f"Leyendo archivo de tracking: {tracking_file}")
    df = pd.read_csv(tracking_file)
    
    # Parsear las listas de ganancias
    df['ganancias_parsed'] = df['ganancias_list'].apply(parse_list_string)
    
    # Ordenar por ganancia promedio de mayor a menor
    df_sorted = df.sort_values('ganancia_mean', ascending=False).reset_index(drop=True)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Análisis de {len(df_sorted)} experimentos")
    logger.info(f"{'='*80}")
    
    # El primer modelo es el mejor
    best_model = df_sorted.iloc[0]
    
    logger.info(f"\n🏆 MEJOR MODELO (Primera posición)")
    logger.info(f"  Version: {best_model['version']}")
    logger.info(f"  Name: {best_model['experiment_name']}")
    logger.info(f"  Tag: {best_model['experiment_tag']}")
    logger.info(f"  Ganancia promedio: {best_model['ganancia_mean']:,.0f}")
    logger.info(f"  Sends promedio: {best_model['n_sends_mean']:,.2f}")
    
    # Obtener las ganancias del mejor modelo
    best_ganancias = best_model['ganancias_parsed']
    
    # Aplicar test de Wilcoxon con los demás modelos
    non_significant_models = []
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Comparación con Wilcoxon (alpha={alpha})")
    logger.info(f"{'='*80}")
    
    for idx in range(1, len(df_sorted)):
        other_model = df_sorted.iloc[idx]
        other_ganancias = other_model['ganancias_parsed']
        
        # Aplicar test de Wilcoxon
        try:
            # Verificar que ambas listas tengan el mismo tamaño
            if len(best_ganancias) != len(other_ganancias):
                logger.warning(f"\n  Comparación #{idx+1}: {other_model['experiment_name']}")
                logger.warning(f"    Tag: {other_model['experiment_tag']}")
                logger.warning(f"    ⚠️ Tamaños diferentes de listas: {len(best_ganancias)} vs {len(other_ganancias)}")
                logger.warning(f"    No se puede aplicar test de Wilcoxon")
                continue
            
            statistic, p_value = wilcoxon(best_ganancias, other_ganancias, 
                                         alternative='greater')
            
            is_significant = p_value < alpha
            
            logger.info(f"\n  Comparación #{idx+1}: {other_model['experiment_name']}")
            logger.info(f"    Tag: {other_model['experiment_tag']}")
            logger.info(f"    Ganancia promedio: {other_model['ganancia_mean']:,.0f}")
            logger.info(f"    Sends promedio: {other_model['n_sends_mean']:,.2f}")
            logger.info(f"    P-value: {p_value:.4f}")
            logger.info(f"    Diferencia significativa: {'Sí' if is_significant else 'No ❌'}")
            
            if not is_significant:
                non_significant_models.append({
                    'version': other_model['version'],
                    'name': other_model['experiment_name'],
                    'tag': other_model['experiment_tag'],
                    'ganancia_mean': other_model['ganancia_mean'],
                    'n_sends_mean': other_model['n_sends_mean'],
                    'p_value': p_value
                })
        except Exception as e:
            logger.warning(f"\n  Comparación #{idx+1}: {other_model['experiment_name']}")
            logger.warning(f"    Error en test de Wilcoxon: {e}")
            continue
    
    # Loggear modelos no significativamente diferentes
    if non_significant_models:
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 Modelos SIN diferencia significativa con el mejor:")
        logger.info(f"{'='*80}")
        
        for i, model in enumerate(non_significant_models, 1):
            logger.info(f"\n  Modelo #{i}")
            logger.info(f"    Version: {model['version']}")
            logger.info(f"    Name: {model['name']}")
            logger.info(f"    Tag: {model['tag']}")
            logger.info(f"    Ganancia promedio: {model['ganancia_mean']:,.0f}")
            logger.info(f"    Sends promedio: {model['n_sends_mean']:,.2f}")
            logger.info(f"    P-value: {model['p_value']:.4f}")
    else:
        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Todos los demás modelos son significativamente inferiores al mejor")
        logger.info(f"{'='*80}")
    
    logger.info("\n")


def compare_with_best_model(ganancias, tracking_file=None, alpha=0.05):
    """
    Toma una lista de ganancias y la compara con el mejor modelo del archivo de tracking
    usando test de Wilcoxon.
    
    Parameters:
    -----------
    ganancias : list or array-like
        Lista de ganancias obtenidas
    tracking_file : str, optional
        Ruta al archivo CSV con los resultados de los experimentos.
        Si no se proporciona, usa 'results/experiments_tracking.csv'
    alpha : float, default=0.05
        Nivel de significancia para el test de Wilcoxon
        
    Returns:
    --------
    dict
        Diccionario con los resultados de la comparación:
        - 'is_better': bool, si las ganancias son significativamente mejores
        - 'p_value': float, p-value del test
        - 'best_model': dict, información del mejor modelo del tracking
        - 'comparison_mean': float, ganancia promedio de la nueva lista
        - 'best_mean': float, ganancia promedio del mejor modelo
    """
    if tracking_file is None:
        tracking_file = 'results/experiments_tracking.csv'
    # Convertir a lista de Python
    ganancias = list(np.array(ganancias))
    
    # Validar que el archivo existe
    if not os.path.exists(tracking_file):
        logger.error(f"❌ El archivo de tracking no existe: {tracking_file}")
        logger.error("No se puede realizar la comparación. Asegúrate de haber ejecutado experimentos primero.")
        return {
            'error': 'Archivo de tracking no existe',
            'is_better': None,
            'p_value': None
        }
    
    # Leer el archivo de tracking
    logger.info(f"Leyendo archivo de tracking: {tracking_file}")
    try:
        df = pd.read_csv(tracking_file)
    except Exception as e:
        logger.error(f"❌ Error al leer el archivo de tracking: {e}")
        return {
            'error': f'Error al leer archivo: {str(e)}',
            'is_better': None,
            'p_value': None
        }
    
    # Validar que el DataFrame no está vacío
    if df.empty:
        logger.error(f"❌ El archivo de tracking está vacío: {tracking_file}")
        logger.error("No hay modelos disponibles para comparar.")
        return {
            'error': 'Archivo de tracking vacío',
            'is_better': None,
            'p_value': None
        }
    
    # Parsear las listas de ganancias
    df['ganancias_parsed'] = df['ganancias_list'].apply(parse_list_string)
    
    # Validar que hay modelos con listas de ganancias válidas (no vacías)
    valid_models = df[df['ganancias_parsed'].apply(lambda x: len(x) > 0)]
    
    if valid_models.empty:
        logger.error("❌ No hay modelos con datos de ganancias válidos en el tracking")
        logger.error("Todos los modelos tienen listas de ganancias vacías.")
        return {
            'error': 'No hay modelos con datos válidos',
            'is_better': None,
            'p_value': None
        }
    
    # Usar solo modelos válidos
    df = valid_models.reset_index(drop=True)
    
    # Ordenar por ganancia promedio de mayor a menor
    df_sorted = df.sort_values('ganancia_mean', ascending=False).reset_index(drop=True)
    
    # Obtener el mejor modelo
    best_model = df_sorted.iloc[0]
    best_ganancias = best_model['ganancias_parsed']
    
    # Calcular estadísticas descriptivas
    nueva_media = np.mean(ganancias)
    mejor_media = best_model['ganancia_mean']
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Comparación de nuevas ganancias con el mejor modelo")
    logger.info(f"{'='*80}")
    logger.info(f"\n📊 Estadísticas de las nuevas ganancias:")
    logger.info(f"  Media: {nueva_media:,.0f}")
    logger.info(f"  Median: {np.median(ganancias):,.0f}")
    logger.info(f"  Std: {np.std(ganancias):,.0f}")
    
    logger.info(f"\n🏆 Mejor modelo del tracking:")
    logger.info(f"  Version: {best_model['version']}")
    logger.info(f"  Name: {best_model['experiment_name']}")
    logger.info(f"  Tag: {best_model['experiment_tag']}")
    logger.info(f"  Media: {mejor_media:,.0f}")
    logger.info(f"  Median: {np.median(best_ganancias):,.0f}")
    logger.info(f"  Std: {np.std(best_ganancias):,.0f}")
    
    # Aplicar test de Wilcoxon
    resultado = {
        'comparison_mean': nueva_media,
        'best_mean': mejor_media,
        'best_model': {
            'version': best_model['version'],
            'name': best_model['experiment_name'],
            'tag': best_model['experiment_tag']
        }
    }
    
    try:
        # Verificar que ambas listas tengan el mismo tamaño
        if len(ganancias) != len(best_ganancias):
            logger.warning(f"\n⚠️ Tamaños diferentes: {len(ganancias)} vs {len(best_ganancias)}")
            logger.warning(f"⚠️ No se puede aplicar test de Wilcoxon")
            resultado['is_better'] = None
            resultado['p_value'] = None
            resultado['error'] = "Tamaños diferentes de listas"
            return resultado
        
        # Test de Wilcoxon - ¿las nuevas ganancias son mayores que las del mejor modelo?
        statistic, p_value = wilcoxon(ganancias, best_ganancias, alternative='greater')
        
        is_better = p_value < alpha
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Resultado del test de Wilcoxon (alpha={alpha})")
        logger.info(f"{'='*80}")
        logger.info(f"  Statistic: {statistic:.4f}")
        logger.info(f"  P-value: {p_value:.4f}")
        logger.info(f"  Diferencia en media: {(nueva_media - mejor_media):,.0f}")
        
        if is_better:
            logger.info(f"  ✅ Las nuevas ganancias son SIGNIFICATIVAMENTE MEJORES que el mejor modelo")
        elif p_value > (1 - alpha):
            # Esto no es directamente posible con alternative='greater', pero lo verificamos
            logger.info(f"  ❌ Las nuevas ganancias NO son mejores que el mejor modelo")
        else:
            logger.info(f"  ➡️ Las nuevas ganancias NO son significativamente diferentes")
        
        resultado['is_better'] = is_better
        resultado['p_value'] = p_value
        resultado['statistic'] = statistic
        resultado['mean_difference'] = nueva_media - mejor_media
        
    except Exception as e:
        logger.error(f"\n❌ Error en test de Wilcoxon: {e}")
        resultado['is_better'] = None
        resultado['p_value'] = None
        resultado['error'] = str(e)
    
    logger.info(f"\n")
    
    return resultado

