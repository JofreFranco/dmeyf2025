#!/usr/bin/env python3
"""
Script para probar la paralelización de transformers.
Prueba PercentileTransformer con paralelización por foto_mes
y TendencyTransformer con paralelización por numero_de_cliente.
"""

import pandas as pd
import numpy as np
import time
import logging
from src.dmeyf2025.processors.feature_processors import PercentileTransformer, TendencyTransformer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data(use_full_dataset=True, sample_fraction=0.05, n_clientes=1000, n_meses=12):
    """
    Carga datos de muestra o genera datos sintéticos para pruebas.
    
    Parameters:
    -----------
    use_full_dataset : bool
        Si True, usa todo el dataset. Si False, toma una muestra
    sample_fraction : float
        Fracción del dataset a usar cuando use_full_dataset=True (por ejemplo, 0.05 = 5%)
    n_clientes : int
        Número de clientes a generar (solo si use_full_dataset=False)
    n_meses : int
        Número de meses a generar
        
    Returns:
    --------
    pd.DataFrame
    """
    logger.info("Intentando cargar datos reales...")
    
    try:
        # Intentar cargar datos reales
        df = pd.read_csv('data/competencia_01_crudo.csv')
        logger.info(f"Datos reales cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        if use_full_dataset:
            if sample_fraction < 1.0:
                # Tomar una muestra aleatoria estratificada por foto_mes
                logger.info(f"Tomando muestra del {sample_fraction*100:.1f}% del dataset...")
                df_sample = df.groupby('foto_mes', group_keys=False).apply(
                    lambda x: x.sample(frac=sample_fraction, random_state=42)
                ).reset_index(drop=True)
                logger.info(f"Muestra creada: {df_sample.shape[0]} filas, {len(df_sample['numero_de_cliente'].unique())} clientes únicos, {df_sample['foto_mes'].nunique()} meses")
                return df_sample
            else:
                logger.info(f"Usando DATASET COMPLETO: {df.shape[0]} filas, {len(df['numero_de_cliente'].unique())} clientes únicos")
                return df
        else:
            # Tomar una muestra para las pruebas
            clientes_sample = df['numero_de_cliente'].unique()[:n_clientes]
            df_sample = df[df['numero_de_cliente'].isin(clientes_sample)].copy()
            
            logger.info(f"Muestra creada: {df_sample.shape[0]} filas, {len(df_sample['numero_de_cliente'].unique())} clientes")
            return df_sample
        
    except FileNotFoundError:
        logger.warning("No se encontraron datos reales. Generando datos sintéticos...")
        
        # Generar datos sintéticos si no hay datos reales
        np.random.seed(42)
        
        # Generar combinaciones de cliente x mes
        foto_mes_values = [202001 + i for i in range(n_meses)]
        cliente_ids = [100000 + i for i in range(n_clientes)]
        
        data = []
        for cliente in cliente_ids:
            for mes in foto_mes_values:
                data.append({
                    'numero_de_cliente': cliente,
                    'foto_mes': mes,
                    'mrentabilidad': np.random.randn() * 1000,
                    'mactivos_margen': np.random.exponential(5000),
                    'mpasivos_margen': np.random.exponential(3000),
                    'mcuentas_saldo': np.random.exponential(10000),
                    'cproductos': np.random.randint(1, 20),
                    'cliente_antiguedad': np.random.randint(1, 100),
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Datos sintéticos generados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df


def test_percentile_transformer(df, n_jobs=4):
    """
    Prueba PercentileTransformer con y sin paralelización por foto_mes.
    """
    logger.info("\n" + "="*80)
    logger.info("PRUEBA: PercentileTransformer")
    logger.info("="*80)
    
    # Variables numéricas para testear
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                    if col not in ['foto_mes', 'numero_de_cliente']]
    
    logger.info(f"Variables numéricas a transformar: {numeric_cols[:5]}...")
    logger.info(f"Total de filas: {len(df)}, Total de meses: {df['foto_mes'].nunique()}")
    
    # Crear transformer
    transformer = PercentileTransformer(
        variables=numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols,  # Limitar para testing
        exclude_cols=['foto_mes', 'numero_de_cliente']
    )
    
    # Fit del transformer
    logger.info("Haciendo fit del transformer...")
    transformer.fit(df)
    
    # Prueba 1: Sin paralelización
    logger.info("\n--- Prueba 1: SIN paralelización ---")
    df_copy = df.copy()
    start_time = time.time()
    df_result_serial = transformer.transform(df_copy, parallel=False)
    time_serial = time.time() - start_time
    logger.info(f"Tiempo sin paralelización: {time_serial:.2f} segundos")
    logger.info(f"Forma resultante: {df_result_serial.shape}")
    
    # Prueba 2: Con paralelización por foto_mes
    logger.info("\n--- Prueba 2: CON paralelización por foto_mes ---")
    df_copy = df.copy()
    start_time = time.time()
    df_result_parallel_foto = transformer.transform(df_copy, parallel=True, parallelize_by='foto_mes', n_jobs=n_jobs)
    time_parallel_foto = time.time() - start_time
    logger.info(f"Tiempo con paralelización por foto_mes: {time_parallel_foto:.2f} segundos")
    logger.info(f"Forma resultante: {df_result_parallel_foto.shape}")
    
    # Comparar resultados
    speedup_foto = time_serial / time_parallel_foto
    logger.info(f"\nSpeedup (foto_mes): {speedup_foto:.2f}x")
    logger.info(f"Mejora (foto_mes): {((time_serial - time_parallel_foto) / time_serial * 100):.1f}%")
    
    # Prueba 3: Con paralelización por columna
    logger.info("\n--- Prueba 3: CON paralelización por columna ---")
    df_copy = df.copy()
    start_time = time.time()
    df_result_parallel_col = transformer.transform(df_copy, parallel=True, parallelize_by='column', n_jobs=n_jobs)
    time_parallel_col = time.time() - start_time
    logger.info(f"Tiempo con paralelización por columna: {time_parallel_col:.2f} segundos")
    logger.info(f"Forma resultante: {df_result_parallel_col.shape}")
    
    # Comparar resultados
    speedup_col = time_serial / time_parallel_col
    logger.info(f"\nSpeedup (column): {speedup_col:.2f}x")
    logger.info(f"Mejora (column): {((time_serial - time_parallel_col) / time_serial * 100):.1f}%")
    
    # Verificar que los resultados son idénticos
    logger.info("\n--- Verificación de resultados (foto_mes vs serial) ---")
    for col in transformer.selected_variables_[:5]:  # Solo primeras 5 para no saturar
        if col in df_result_serial.columns and col in df_result_parallel_foto.columns:
            # Ordenar por índice para comparar
            serial_sorted = df_result_serial[col].sort_index()
            parallel_sorted = df_result_parallel_foto[col].sort_index()
            
            # Comparar con tolerancia para floats
            are_equal = np.allclose(
                serial_sorted.fillna(0), 
                parallel_sorted.fillna(0), 
                rtol=1e-5, 
                atol=1e-8
            )
            
            if not are_equal:
                logger.warning(f"❌ Diferencias encontradas en columna: {col}")
                diff = np.abs(serial_sorted.fillna(0) - parallel_sorted.fillna(0))
                logger.warning(f"   Max diferencia: {diff.max()}")
            else:
                logger.info(f"✓ Columna {col}: foto_mes = serial")
    
    logger.info("\n--- Verificación de resultados (column vs serial) ---")
    for col in transformer.selected_variables_[:5]:  # Solo primeras 5 para no saturar
        if col in df_result_serial.columns and col in df_result_parallel_col.columns:
            # Ordenar por índice para comparar
            serial_sorted = df_result_serial[col].sort_index()
            parallel_sorted = df_result_parallel_col[col].sort_index()
            
            # Comparar con tolerancia para floats
            are_equal = np.allclose(
                serial_sorted.fillna(0), 
                parallel_sorted.fillna(0), 
                rtol=1e-5, 
                atol=1e-8
            )
            
            if not are_equal:
                logger.warning(f"❌ Diferencias encontradas en columna: {col}")
                diff = np.abs(serial_sorted.fillna(0) - parallel_sorted.fillna(0))
                logger.warning(f"   Max diferencia: {diff.max()}")
            else:
                logger.info(f"✓ Columna {col}: column = serial")
    
    return {
        'time_serial': time_serial,
        'time_parallel_foto': time_parallel_foto,
        'time_parallel_col': time_parallel_col,
        'speedup_foto': speedup_foto,
        'speedup_col': speedup_col
    }


def test_tendency_transformer(df, n_jobs=4):
    """
    Prueba TendencyTransformer con y sin paralelización por numero_de_cliente.
    """
    logger.info("\n" + "="*80)
    logger.info("PRUEBA: TendencyTransformer")
    logger.info("="*80)
    
    # Asegurarse de que hay columnas 'm...'
    m_cols = [col for col in df.columns if col.startswith('m') and col not in ['foto_mes']]
    
    if not m_cols:
        logger.warning("No hay columnas que empiecen con 'm'. Creando algunas...")
        df = df.copy()
        df['mvariable1'] = np.random.randn(len(df)) * 1000
        df['mvariable2'] = np.random.randn(len(df)) * 500
        m_cols = ['mvariable1', 'mvariable2']
    
    logger.info(f"Variables 'm' a transformar: {m_cols[:5]}...")
    logger.info(f"Total de filas: {len(df)}, Total de clientes: {df['numero_de_cliente'].nunique()}")
    
    # Crear transformer
    transformer = TendencyTransformer(
        exclude_cols=['foto_mes', 'numero_de_cliente'],
        window=6
    )
    
    # Fit del transformer
    logger.info("Haciendo fit del transformer...")
    transformer.fit(df)
    
    # Prueba 1: Sin paralelización
    logger.info("\n--- Prueba 1: SIN paralelización ---")
    df_copy = df.copy()
    start_time = time.time()
    df_result_serial = transformer.transform(df_copy, parallel=False)
    time_serial = time.time() - start_time
    logger.info(f"Tiempo sin paralelización: {time_serial:.2f} segundos")
    logger.info(f"Forma resultante: {df_result_serial.shape}")
    
    # Prueba 2: Con paralelización por numero_de_cliente
    logger.info("\n--- Prueba 2: CON paralelización por numero_de_cliente ---")
    df_copy = df.copy()
    start_time = time.time()
    df_result_parallel = transformer.transform(df_copy, parallel=True, parallelize_by='numero_cliente', n_jobs=n_jobs)
    time_parallel = time.time() - start_time
    logger.info(f"Tiempo con paralelización: {time_parallel:.2f} segundos")
    logger.info(f"Forma resultante: {df_result_parallel.shape}")
    
    # Comparar resultados
    speedup = time_serial / time_parallel
    logger.info(f"\nSpeedup: {speedup:.2f}x")
    logger.info(f"Mejora: {((time_serial - time_parallel) / time_serial * 100):.1f}%")
    
    # Verificar que los resultados son idénticos
    logger.info("\n--- Verificación de resultados ---")
    tendency_cols = [col for col in df_result_serial.columns if '_tendency' in col]
    
    for col in tendency_cols[:5]:  # Verificar las primeras 5 columnas
        if col in df_result_serial.columns and col in df_result_parallel.columns:
            # Ordenar por índice para comparar
            serial_sorted = df_result_serial[col].sort_index()
            parallel_sorted = df_result_parallel[col].sort_index()
            
            # Comparar con tolerancia para floats
            are_equal = np.allclose(
                serial_sorted.fillna(0), 
                parallel_sorted.fillna(0), 
                rtol=1e-5, 
                atol=1e-8
            )
            
            if not are_equal:
                logger.warning(f"❌ Diferencias encontradas en columna: {col}")
                diff = np.abs(serial_sorted.fillna(0) - parallel_sorted.fillna(0))
                logger.warning(f"   Max diferencia: {diff.max()}")
            else:
                logger.info(f"✓ Columna {col}: Resultados idénticos")
    
    return {
        'time_serial': time_serial,
        'time_parallel': time_parallel,
        'speedup': speedup
    }


def main():
    """
    Función principal para ejecutar las pruebas.
    """
    logger.info("="*80)
    logger.info("INICIANDO PRUEBAS DE PARALELIZACIÓN")
    logger.info("="*80)
    
    # Cargar datos - USAR 50% DEL DATASET
    df = load_sample_data(use_full_dataset=True, sample_fraction=0.50)
    
    # Configurar número de jobs (-1 = todos los cores disponibles)
    n_jobs = -1
    logger.info(f"\nUsando n_jobs={n_jobs} (todos los cores disponibles) para paralelización")
    
    # Ejecutar pruebas
    results = {}
    
    try:
        results['percentile'] = test_percentile_transformer(df, n_jobs=n_jobs)
    except Exception as e:
        logger.error(f"Error en PercentileTransformer: {e}", exc_info=True)
    
    try:
        results['tendency'] = test_tendency_transformer(df, n_jobs=n_jobs)
    except Exception as e:
        logger.error(f"Error en TendencyTransformer: {e}", exc_info=True)
    
    # Resumen final
    logger.info("\n" + "="*80)
    logger.info("RESUMEN DE RESULTADOS")
    logger.info("="*80)
    
    if 'percentile' in results:
        r = results['percentile']
        logger.info(f"\nPercentileTransformer:")
        logger.info(f"  Tiempo serial:           {r['time_serial']:.2f}s")
        logger.info(f"  Tiempo paralelo (foto):  {r['time_parallel_foto']:.2f}s  | Speedup: {r['speedup_foto']:.2f}x")
        logger.info(f"  Tiempo paralelo (column): {r['time_parallel_col']:.2f}s  | Speedup: {r['speedup_col']:.2f}x")
    
    if 'tendency' in results:
        r = results['tendency']
        logger.info(f"\nTendencyTransformer (por numero_cliente):")
        logger.info(f"  Tiempo serial:   {r['time_serial']:.2f}s")
        logger.info(f"  Tiempo paralelo: {r['time_parallel']:.2f}s")
        logger.info(f"  Speedup:         {r['speedup']:.2f}x")
    
    logger.info("\n" + "="*80)
    logger.info("PRUEBAS COMPLETADAS")
    logger.info("="*80)


if __name__ == "__main__":
    main()

