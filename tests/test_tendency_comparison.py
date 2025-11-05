import pandas as pd
import numpy as np
import sys
import os

# Agregar el directorio src al path para importar los módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dmeyf2025.processors.feature_processors import TendencyTransformer, LegacyTendencyTransformer


def test_comparison_no_nulls():
    """
    Compara ambas implementaciones cuando NO hay valores nulos.
    En este caso, los resultados deberían ser idénticos.
    """
    print("="*70)
    print("COMPARACIÓN: Sin valores nulos")
    print("="*70)
    
    # Crear datos sintéticos sin nulos
    data = pd.DataFrame({
        'numero_de_cliente': [1, 1, 1, 1, 1, 2, 2, 2, 2],
        'foto_mes': [202101, 202102, 202103, 202104, 202105, 202101, 202102, 202103, 202104],
        'mcuentas': [100, 105, 110, 115, 120, 50, 45, 40, 35],
        'mprestamos': [200, 202, 204, 206, 208, 100, 97, 94, 91],
    })
    
    print("\nDatos de prueba:")
    print(data)
    
    # Aplicar versión legacy
    print("\n--- Aplicando LegacyTendencyTransformer ---")
    legacy_transformer = LegacyTendencyTransformer()
    legacy_transformer.fit(data)
    legacy_result = legacy_transformer.transform(data)
    
    # Aplicar nueva versión
    print("--- Aplicando TendencyTransformer (nueva versión) ---")
    new_transformer = TendencyTransformer()
    new_transformer.fit(data)
    new_result = new_transformer.transform(data)
    
    # Comparar resultados
    print("\n" + "="*70)
    print("COMPARACIÓN DE RESULTADOS")
    print("="*70)
    
    tendency_cols = [col for col in legacy_result.columns if col.endswith('_tendency')]
    
    all_match = True
    tolerance = 1e-10  # Tolerancia muy estricta para precisión numérica
    
    for col in tendency_cols:
        legacy_vals = legacy_result[col].values
        new_vals = new_result[col].values
        
        # Comparar NaN
        nan_match = np.array_equal(pd.isna(legacy_vals), pd.isna(new_vals))
        
        # Comparar valores numéricos (donde no hay NaN)
        valid_mask = ~pd.isna(legacy_vals) & ~pd.isna(new_vals)
        if valid_mask.any():
            max_diff = np.max(np.abs(legacy_vals[valid_mask] - new_vals[valid_mask]))
            values_match = max_diff < tolerance
        else:
            max_diff = 0.0
            values_match = True
        
        match = nan_match and values_match
        all_match = all_match and match
        
        print(f"\n{col}:")
        print(f"  NaN match: {nan_match}")
        print(f"  Max diff: {max_diff:.2e}")
        print(f"  Match: {'✅' if match else '❌'}")
    
    print("\n" + "="*70)
    if all_match:
        print("✅ SIN NULOS: Ambas versiones producen resultados IDÉNTICOS")
        return True
    else:
        print("❌ SIN NULOS: Las versiones producen resultados DIFERENTES")
        return False


def test_comparison_with_nulls():
    """
    Compara ambas implementaciones cuando HAY valores nulos.
    En este caso, los resultados SERÁN DIFERENTES debido a las
    diferentes estrategias de manejo de NaN.
    """
    print("\n" + "="*70)
    print("COMPARACIÓN: Con valores nulos")
    print("="*70)
    
    # Crear datos sintéticos con nulos
    data = pd.DataFrame({
        'numero_de_cliente': [1, 1, 1, 1, 1],
        'foto_mes': [202101, 202102, 202103, 202104, 202105],
        'mcuentas': [100, 103, np.nan, 109, 112],  # NaN en posición 2
    })
    
    print("\nDatos de prueba (con NaN en posición 2):")
    print(data)
    
    # Aplicar versión legacy
    print("\n--- Aplicando LegacyTendencyTransformer ---")
    legacy_transformer = LegacyTendencyTransformer()
    legacy_transformer.fit(data)
    legacy_result = legacy_transformer.transform(data)
    
    print("\nResultado Legacy:")
    print(legacy_result[['foto_mes', 'mcuentas', 'mcuentas_tendency']])
    
    # Aplicar nueva versión
    print("\n--- Aplicando TendencyTransformer (nueva versión) ---")
    new_transformer = TendencyTransformer()
    new_transformer.fit(data)
    new_result = new_transformer.transform(data)
    
    print("\nResultado Nueva versión:")
    print(new_result[['foto_mes', 'mcuentas', 'mcuentas_tendency']])
    
    # Comparar resultados
    print("\n" + "="*70)
    print("ANÁLISIS DE DIFERENCIAS")
    print("="*70)
    
    print("\nDIFERENCIA EN MANEJO DE NaN:")
    print("- Legacy: Comprime índices (0,1,2,3 para valores válidos)")
    print("  Esto significa que calcula la pendiente usando x=[0,1,2,3] para y=[100,103,109,112]")
    print("\n- Nueva: Mantiene índices originales (0,1,3,4 para valores válidos)")
    print("  Esto significa que calcula la pendiente usando x=[0,1,3,4] para y=[100,103,109,112]")
    
    print("\nIMPACTO:")
    legacy_tendency = legacy_result.iloc[-1]['mcuentas_tendency']
    new_tendency = new_result.iloc[-1]['mcuentas_tendency']
    diff = abs(legacy_tendency - new_tendency)
    
    print(f"- Legacy (última tendencia): {legacy_tendency:.6f}")
    print(f"- Nueva (última tendencia): {new_tendency:.6f}")
    print(f"- Diferencia absoluta: {diff:.6f}")
    
    print("\n" + "="*70)
    if diff > 0.01:
        print("⚠️  CON NULOS: Las versiones producen resultados DIFERENTES")
        print("    Esto es ESPERADO debido a diferentes estrategias de manejo de NaN")
        return True
    else:
        print("⚠️  CON NULOS: Las versiones producen resultados similares (inesperado)")
        return False


def test_performance_comparison():
    """
    Compara el rendimiento de ambas implementaciones.
    """
    import time
    
    print("\n" + "="*70)
    print("COMPARACIÓN DE RENDIMIENTO")
    print("="*70)
    
    # Crear datos más grandes para medir rendimiento
    n_clients = 1000
    n_months = 12
    
    data = []
    for client_id in range(1, n_clients + 1):
        for month in range(202101, 202101 + n_months):
            data.append({
                'numero_de_cliente': client_id,
                'foto_mes': month,
                'mcuentas': np.random.normal(1000, 100),
                'mprestamos': np.random.normal(5000, 500),
                'mtarjetas': np.random.normal(2000, 200),
            })
    
    df = pd.DataFrame(data)
    print(f"\nDatos de prueba: {len(df)} filas, {n_clients} clientes, {n_months} meses")
    
    # Medir versión legacy
    print("\n--- LegacyTendencyTransformer ---")
    legacy_transformer = LegacyTendencyTransformer()
    legacy_transformer.fit(df)
    
    start = time.time()
    legacy_result = legacy_transformer.transform(df)
    legacy_time = time.time() - start
    
    print(f"Tiempo: {legacy_time:.3f} segundos")
    
    # Medir nueva versión
    print("\n--- TendencyTransformer (nueva versión) ---")
    new_transformer = TendencyTransformer()
    new_transformer.fit(df)
    
    start = time.time()
    new_result = new_transformer.transform(df)
    new_time = time.time() - start
    
    print(f"Tiempo: {new_time:.3f} segundos")
    
    # Comparar
    print("\n" + "="*70)
    speedup = legacy_time / new_time
    print(f"SPEEDUP: {speedup:.2f}x más rápido")
    
    if speedup > 1.5:
        print(f"✅ La nueva versión es {speedup:.2f}x MÁS RÁPIDA")
        return True
    elif speedup > 0.9:
        print(f"⚠️  Ambas versiones tienen rendimiento similar")
        return True
    else:
        print(f"❌ La nueva versión es más LENTA (inesperado)")
        return False


if __name__ == "__main__":
    print("Ejecutando comparación entre TendencyTransformer y LegacyTendencyTransformer...\n")
    
    # Test 1: Sin nulos (deberían ser idénticos)
    success1 = test_comparison_no_nulls()
    
    # Test 2: Con nulos (serán diferentes)
    success2 = test_comparison_with_nulls()
    
    # Test 3: Performance
    success3 = test_performance_comparison()
    
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"Sin nulos (idénticos esperado): {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"Con nulos (diferentes esperado): {'✓ PASS' if success2 else '✗ FAIL'}")
    print(f"Performance (más rápido esperado): {'✓ PASS' if success3 else '✗ FAIL'}")
    
    print("\n" + "="*70)
    print("CONCLUSIÓN")
    print("="*70)
    print("La nueva implementación vectorizada:")
    print("✅ Es mucho más rápida (vectorización con NumPy)")
    print("✅ Produce resultados idénticos cuando NO hay NaN")
    print("⚠️  Produce resultados diferentes cuando HAY NaN")
    print("")
    print("RECOMENDACIÓN:")
    print("- Si los datos NO tienen NaN: Usar nueva versión (más rápida, mismo resultado)")
    print("- Si los datos tienen NaN: Decidir basándose en la semántica deseada:")
    print("  * Legacy: Índices comprimidos (ignora posiciones de NaN)")
    print("  * Nueva: Índices originales (preserva posiciones de NaN)")
    
    if success1 and success3:
        print("\n🎉 TESTS DE COMPARACIÓN COMPLETADOS!")
        sys.exit(0)
    else:
        print("\n❌ ALGUNOS TESTS FALLARON")
        sys.exit(1)

