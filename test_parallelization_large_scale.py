#!/usr/bin/env python3
"""
Script para probar la paralelización de PercentileTransformer
con un escenario realista: 1000+ columnas y 30 meses.
"""

import pandas as pd
import numpy as np
import time
import logging
from src.dmeyf2025.processors.feature_processors import PercentileTransformer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_large_dataset(n_rows=50000, n_cols=1000, n_months=30):
    """
    Crea un dataset sintético grande para simular el escenario real.
    
    Parameters:
    -----------
    n_rows : int
        Número de filas por mes (total será n_rows * n_months)
    n_cols : int
        Número de columnas numéricas a generar
    n_months : int
        Número de meses diferentes
    """
    logger.info(f"Generando dataset sintético: {n_rows} filas/mes x {n_months} meses x {n_cols} columnas...")
    
    np.random.seed(42)
    
    # Generar meses (desde 202001)
    foto_mes_values = [202001 + i for i in range(n_months)]
    
    # Crear dataframe base
    data = {
        'foto_mes': np.repeat(foto_mes_values, n_rows),
        'numero_de_cliente': np.tile(np.arange(1000000, 1000000 + n_rows), n_months)
    }
    
    df = pd.DataFrame(data)
    
    # Generar columnas numéricas con diferentes distribuciones
    logger.info(f"Generando {n_cols} columnas numéricas...")
    for i in range(n_cols):
        if i % 100 == 0:
            logger.info(f"  Generadas {i}/{n_cols} columnas...")
        
        # Variar el tipo de distribución
        if i % 4 == 0:
            # Distribución normal
            df[f'var_{i:04d}'] = np.random.randn(len(df)) * 1000
        elif i % 4 == 1:
            # Distribución exponencial (con ceros)
            values = np.random.exponential(1000, len(df))
            values[np.random.rand(len(df)) < 0.2] = 0  # 20% ceros
            df[f'var_{i:04d}'] = values
        elif i % 4 == 2:
            # Distribución uniforme
            df[f'var_{i:04d}'] = np.random.uniform(-500, 500, len(df))
        else:
            # Distribución con muchos valores negativos
            df[f'var_{i:04d}'] = np.random.randn(len(df)) * 500 - 200
        
        # Agregar algunos NaN (5%)
        nan_mask = np.random.rand(len(df)) < 0.05
        df.loc[nan_mask, f'var_{i:04d}'] = np.nan
    
    logger.info(f"Dataset generado: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")
    logger.info(f"Tamaño en memoria: ~{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return df


def test_percentile_large_scale(n_rows=50000, n_cols=1000, n_months=30, n_jobs=-1):
    """
    Prueba PercentileTransformer con dataset grande.
    """
    logger.info("\n" + "="*80)
    logger.info("PRUEBA: PercentileTransformer con Dataset Grande")
    logger.info("="*80)
    
    # Crear dataset
    df = create_large_dataset(n_rows=n_rows, n_cols=n_cols, n_months=n_months)
    
    # Obtener columnas numéricas (todas las var_XXXX)
    numeric_cols = [col for col in df.columns if col.startswith('var_')]
    
    logger.info(f"\nConfiguración:")
    logger.info(f"  Total filas: {len(df):,}")
    logger.info(f"  Total meses: {n_months}")
    logger.info(f"  Columnas numéricas: {len(numeric_cols)}")
    logger.info(f"  N_jobs: {n_jobs}")
    
    # Crear transformer
    transformer = PercentileTransformer(
        variables=numeric_cols,
        exclude_cols=['foto_mes', 'numero_de_cliente']
    )
    
    # Fit del transformer
    logger.info("\nHaciendo fit del transformer...")
    transformer.fit(df)
    
    # Prueba 1: Sin paralelización
    logger.info("\n" + "-"*80)
    logger.info("Prueba 1: SIN paralelización")
    logger.info("-"*80)
    df_copy = df.copy()
    start_time = time.time()
    df_result_serial = transformer.transform(df_copy, parallel=False)
    time_serial = time.time() - start_time
    logger.info(f"✓ Tiempo sin paralelización: {time_serial:.2f} segundos")
    logger.info(f"  Forma resultante: {df_result_serial.shape}")
    
    # Prueba 2: Con paralelización por foto_mes
    logger.info("\n" + "-"*80)
    logger.info("Prueba 2: Paralelización por FOTO_MES")
    logger.info("-"*80)
    df_copy = df.copy()
    start_time = time.time()
    df_result_parallel_foto = transformer.transform(df_copy, parallel=True, parallelize_by='foto_mes', n_jobs=n_jobs)
    time_parallel_foto = time.time() - start_time
    speedup_foto = time_serial / time_parallel_foto
    logger.info(f"✓ Tiempo con paralelización (foto_mes): {time_parallel_foto:.2f} segundos")
    logger.info(f"  Speedup: {speedup_foto:.2f}x")
    logger.info(f"  Mejora: {((time_serial - time_parallel_foto) / time_serial * 100):.1f}%")
    
    # Prueba 3: Con paralelización por columna
    logger.info("\n" + "-"*80)
    logger.info("Prueba 3: Paralelización por COLUMNA")
    logger.info("-"*80)
    df_copy = df.copy()
    start_time = time.time()
    df_result_parallel_col = transformer.transform(df_copy, parallel=True, parallelize_by='column', n_jobs=n_jobs)
    time_parallel_col = time.time() - start_time
    speedup_col = time_serial / time_parallel_col
    logger.info(f"✓ Tiempo con paralelización (column): {time_parallel_col:.2f} segundos")
    logger.info(f"  Speedup: {speedup_col:.2f}x")
    logger.info(f"  Mejora: {((time_serial - time_parallel_col) / time_serial * 100):.1f}%")
    
    # Verificar resultados (muestra pequeña)
    logger.info("\n" + "-"*80)
    logger.info("Verificación de Resultados (muestra)")
    logger.info("-"*80)
    
    sample_cols = numeric_cols[:5]
    all_equal = True
    
    for col in sample_cols:
        # Comparar serial vs paralelo_foto
        equal_foto = np.allclose(
            df_result_serial[col].fillna(0), 
            df_result_parallel_foto[col].fillna(0), 
            rtol=1e-5, 
            atol=1e-8
        )
        
        # Comparar serial vs paralelo_col
        equal_col = np.allclose(
            df_result_serial[col].fillna(0), 
            df_result_parallel_col[col].fillna(0), 
            rtol=1e-5, 
            atol=1e-8
        )
        
        if equal_foto and equal_col:
            logger.info(f"  ✓ {col}: Todos los métodos coinciden")
        else:
            logger.warning(f"  ❌ {col}: Diferencias detectadas")
            all_equal = False
    
    if all_equal:
        logger.info("\n✓ Verificación exitosa: Todos los métodos producen resultados idénticos")
    
    # Resumen
    logger.info("\n" + "="*80)
    logger.info("RESUMEN DE RESULTADOS")
    logger.info("="*80)
    logger.info(f"\nDataset: {len(df):,} filas, {len(numeric_cols)} columnas, {n_months} meses")
    logger.info(f"\nTiempos de ejecución:")
    logger.info(f"  Serial:             {time_serial:.2f}s")
    logger.info(f"  Paralelo (foto):    {time_parallel_foto:.2f}s  →  Speedup: {speedup_foto:.2f}x")
    logger.info(f"  Paralelo (column):  {time_parallel_col:.2f}s  →  Speedup: {speedup_col:.2f}x")
    
    # Determinar mejor estrategia
    if speedup_foto > 1.0 or speedup_col > 1.0:
        if speedup_foto > speedup_col:
            logger.info(f"\n🏆 MEJOR ESTRATEGIA: foto_mes (ahorra {time_serial - time_parallel_foto:.2f}s)")
        else:
            logger.info(f"\n🏆 MEJOR ESTRATEGIA: column (ahorra {time_serial - time_parallel_col:.2f}s)")
    else:
        logger.info(f"\n💡 RECOMENDACIÓN: No usar paralelización para este tamaño de dataset")
    
    logger.info("\n" + "="*80)
    
    return {
        'time_serial': time_serial,
        'time_parallel_foto': time_parallel_foto,
        'time_parallel_col': time_parallel_col,
        'speedup_foto': speedup_foto,
        'speedup_col': speedup_col
    }


def main():
    """
    Función principal.
    """
    logger.info("="*80)
    logger.info("PRUEBA DE PARALELIZACIÓN - ESCENARIO REALISTA")
    logger.info("="*80)
    
    # Escenario 1: Dataset moderado (para prueba rápida)
    logger.info("\n### ESCENARIO 1: Dataset Moderado ###")
    results_moderate = test_percentile_large_scale(
        n_rows=10000,    # 10k filas por mes
        n_cols=100,      # 100 columnas
        n_months=30,     # 30 meses
        n_jobs=-1
    )
    
    # Escenario 2: Dataset grande (más cercano a tu caso real)
    logger.info("\n\n### ESCENARIO 2: Dataset Grande ###")
    results_large = test_percentile_large_scale(
        n_rows=30000,    # 30k filas por mes  
        n_cols=500,      # 500 columnas (mitad de 1000 para no tardar mucho)
        n_months=30,     # 30 meses
        n_jobs=-1
    )
    
    logger.info("\n" + "="*80)
    logger.info("ANÁLISIS FINAL")
    logger.info("="*80)
    logger.info("\nEscenario Moderado (100 cols, 30 meses):")
    logger.info(f"  Mejor speedup: {max(results_moderate['speedup_foto'], results_moderate['speedup_col']):.2f}x")
    
    logger.info("\nEscenario Grande (500 cols, 30 meses):")
    logger.info(f"  Mejor speedup: {max(results_large['speedup_foto'], results_large['speedup_col']):.2f}x")
    
    # Proyección para 1000 columnas
    if results_large['speedup_col'] > 1.0:
        projected_time = results_large['time_serial'] * (1000 / 500)
        projected_parallel = results_large['time_parallel_col'] * (1000 / 500)
        projected_speedup = projected_time / projected_parallel
        
        logger.info("\n📊 PROYECCIÓN para 1000 columnas:")
        logger.info(f"  Tiempo serial estimado:   ~{projected_time:.1f}s")
        logger.info(f"  Tiempo paralelo estimado: ~{projected_parallel:.1f}s")
        logger.info(f"  Speedup estimado:         ~{projected_speedup:.2f}x")
        logger.info(f"  Ahorro estimado:          ~{projected_time - projected_parallel:.1f}s")


if __name__ == "__main__":
    main()

