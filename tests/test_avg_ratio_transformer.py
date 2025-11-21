import pandas as pd
import numpy as np
import sys
import os

# Agregar el directorio src al path para importar los módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dmeyf2025.processors.feature_processors import AvgRatioTransformer


def test_avg_ratio_transformer_basic():
    """
    Test básico de AvgRatioTransformer con datos sintéticos.
    """
    print("="*60)
    print("Test básico de AvgRatioTransformer")
    print("="*60)
    
    # Crear datos sintéticos
    np.random.seed(42)
    
    data = []
    
    # Cliente 1: Valores crecientes
    # Meses: 202101, 202102, 202103, 202104
    # mcuentas_saldo: 100, 110, 120, 130
    for i, month in enumerate(range(202101, 202105)):
        data.append({
            'numero_de_cliente': 1,
            'foto_mes': month,
            'mcuentas_saldo': 100 + 10 * i,
            'mprestamos_total': 200 + 20 * i,
        })
    
    # Cliente 2: Valores decrecientes
    for i, month in enumerate(range(202101, 202105)):
        data.append({
            'numero_de_cliente': 2,
            'foto_mes': month,
            'mcuentas_saldo': 200 - 10 * i,
            'mprestamos_total': 400 - 20 * i,
        })
    
    df = pd.DataFrame(data)
    print(f"\nDatos sintéticos creados: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Clientes: {df['numero_de_cliente'].nunique()}")
    
    print("\nDatos del Cliente 1:")
    print(df[df['numero_de_cliente'] == 1])
    
    # Crear y aplicar el transformador con window de 2 meses
    print("\n" + "="*60)
    print("Aplicando AvgRatioTransformer con months=2...")
    print("="*60)
    
    transformer = AvgRatioTransformer(months=2)
    transformer.fit(df)
    transformed_df = transformer.transform(df)
    
    print(f"\nDatos transformados: {transformed_df.shape[0]} filas, {transformed_df.shape[1]} columnas")
    
    # Calcular cuántas columnas nuevas se crearon
    original_cols = set(df.columns)
    new_cols = sorted([col for col in transformed_df.columns if col not in original_cols])
    print(f"Columnas nuevas creadas: {len(new_cols)}")
    print(f"Nombres: {new_cols}")
    
    # Nota: La implementación actual solo genera columnas de ratio, no de promedio
    print("\nNota: El transformer solo genera columnas _avg_ratio, no columnas _avg")
    
    # Verificaciones
    print("\n" + "="*60)
    print("VERIFICACIONES")
    print("="*60)
    
    all_tests_passed = True
    
    # Test 1: Cliente 1, mes 202103 (tiene 2 meses previos)
    print("\n--- Test 1: Cliente 1, mes 202103 ---")
    client1_month3 = transformed_df[(transformed_df['numero_de_cliente'] == 1) & 
                                     (transformed_df['foto_mes'] == 202103)]
    
    # mcuentas_saldo en 202103 = 120
    # Promedio de 202101 (100) y 202102 (110) = 105
    # Ratio = 120 / 105 = 1.142857...
    
    ratio_value = client1_month3['mcuentas_saldo_avg_ratio_2m'].values[0]
    
    expected_ratio = 120.0 / 105.0  # 1.142857...
    
    print(f"mcuentas_saldo: {client1_month3['mcuentas_saldo'].values[0]}")
    print(f"mcuentas_saldo_avg_ratio_2m: {ratio_value:.6f} (esperado: {expected_ratio:.6f})")
    
    tolerance = 0.001
    test1_pass = abs(ratio_value - expected_ratio) < tolerance
    
    if test1_pass:
        print("✅ Test 1 PASS: Promedio y ratio correctos")
    else:
        print("❌ Test 1 FAIL: Promedio o ratio incorrectos")
        all_tests_passed = False
    
    # Test 2: Cliente 1, mes 202104 (tiene 3 meses previos pero window=2)
    print("\n--- Test 2: Cliente 1, mes 202104 ---")
    client1_month4 = transformed_df[(transformed_df['numero_de_cliente'] == 1) & 
                                     (transformed_df['foto_mes'] == 202104)]
    
    # mcuentas_saldo en 202104 = 130
    # Promedio de los últimos 2 meses: 202102 (110) y 202103 (120) = 115
    # Ratio = 130 / 115 = 1.130434...
    
    ratio_value = client1_month4['mcuentas_saldo_avg_ratio_2m'].values[0]
    
    expected_ratio = 130.0 / 115.0
    
    print(f"mcuentas_saldo: {client1_month4['mcuentas_saldo'].values[0]}")
    print(f"mcuentas_saldo_avg_ratio_2m: {ratio_value:.6f} (esperado: {expected_ratio:.6f})")
    
    test2_pass = abs(ratio_value - expected_ratio) < tolerance
    
    if test2_pass:
        print("✅ Test 2 PASS: Ventana móvil funciona correctamente")
    else:
        print("❌ Test 2 FAIL: Ventana móvil incorrecta")
        all_tests_passed = False
    
    # Test 3: Cliente 2, tendencia decreciente
    print("\n--- Test 3: Cliente 2, mes 202103 (valores decrecientes) ---")
    client2_month3 = transformed_df[(transformed_df['numero_de_cliente'] == 2) & 
                                     (transformed_df['foto_mes'] == 202103)]
    
    # mcuentas_saldo en 202103 = 180
    # Promedio de 202101 (200) y 202102 (190) = 195
    # Ratio = 180 / 195 = 0.923076...
    
    ratio_value = client2_month3['mcuentas_saldo_avg_ratio_2m'].values[0]
    
    expected_ratio = 180.0 / 195.0
    
    print(f"mcuentas_saldo: {client2_month3['mcuentas_saldo'].values[0]}")
    print(f"mcuentas_saldo_avg_ratio_2m: {ratio_value:.6f} (esperado: {expected_ratio:.6f})")
    
    test3_pass = abs(ratio_value - expected_ratio) < tolerance
    
    if test3_pass:
        print("✅ Test 3 PASS: Calcula correctamente para valores decrecientes")
    else:
        print("❌ Test 3 FAIL: Falla con valores decrecientes")
        all_tests_passed = False
    
    # Test 4: Primer mes (sin datos previos)
    print("\n--- Test 4: Primer mes sin datos previos ---")
    client1_month1 = transformed_df[(transformed_df['numero_de_cliente'] == 1) & 
                                     (transformed_df['foto_mes'] == 202101)]
    
    ratio_value = client1_month1['mcuentas_saldo_avg_ratio_2m'].values[0]
    
    print(f"mcuentas_saldo_avg_ratio_2m: {ratio_value} (esperado: inf o valor muy alto)")
    
    # En el primer mes no hay datos previos, por lo que el ratio será infinito o NaN
    test4_pass = pd.isna(ratio_value) or ratio_value > 1e10
    
    if test4_pass:
        print("✅ Test 4 PASS: Primer mes devuelve NaN para el promedio")
    else:
        print("❌ Test 4 FAIL: Primer mes no maneja correctamente falta de datos")
        all_tests_passed = False
    
    # Test 5: Verificar que las columnas originales no se modifican
    print("\n--- Test 5: Integridad de datos originales ---")
    original_data_preserved = df.equals(transformed_df[df.columns])
    
    if original_data_preserved:
        print("✅ Test 5 PASS: Datos originales preservados")
    else:
        print("❌ Test 5 FAIL: Datos originales modificados")
        all_tests_passed = False
    
    # Mostrar resultados completos para Cliente 1
    print("\n--- Resultados completos Cliente 1 ---")
    client1_results = transformed_df[transformed_df['numero_de_cliente'] == 1][
        ['foto_mes', 'mcuentas_saldo', 'mcuentas_saldo_avg_ratio_2m']
    ]
    print(client1_results)
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    
    if all_tests_passed:
        print("🎉 TODOS LOS TESTS PASARON EXITOSAMENTE!")
        return True
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        return False


def test_avg_ratio_transformer_months_3():
    """
    Test con window de 3 meses.
    """
    print("\n" + "="*60)
    print("Test con months=3")
    print("="*60)
    
    data = []
    
    # Cliente con 5 meses de datos
    for i, month in enumerate(range(202101, 202106)):
        data.append({
            'numero_de_cliente': 1,
            'foto_mes': month,
            'mcuentas_saldo': 100 + 10 * i,  # 100, 110, 120, 130, 140
        })
    
    df = pd.DataFrame(data)
    
    transformer = AvgRatioTransformer(months=3)
    transformer.fit(df)
    transformed_df = transformer.transform(df)
    
    # Test: mes 202104 (tiene 3 meses previos completos)
    print("\n--- Test: mes 202104 con 3 meses previos ---")
    month4 = transformed_df[transformed_df['foto_mes'] == 202104]
    
    # mcuentas_saldo en 202104 = 130
    # Promedio de 202101 (100), 202102 (110), 202103 (120) = 110
    # Ratio = 130 / 110 = 1.181818...
    
    ratio_value = month4['mcuentas_saldo_avg_ratio_3m'].values[0]
    
    expected_ratio = 130.0 / 110.0
    
    print(f"mcuentas_saldo: {month4['mcuentas_saldo'].values[0]}")
    print(f"mcuentas_saldo_avg_ratio_3m: {ratio_value:.6f} (esperado: {expected_ratio:.6f})")
    
    tolerance = 0.001
    test_pass = abs(ratio_value - expected_ratio) < tolerance
    
    # Mostrar todos los resultados
    print("\n--- Resultados completos ---")
    results = transformed_df[['foto_mes', 'mcuentas_saldo', 'mcuentas_saldo_avg_ratio_3m']]
    print(results)
    
    if test_pass:
        print("\n✅ Test con months=3 PASS")
        return True
    else:
        print("\n❌ Test con months=3 FAIL")
        return False


def test_avg_ratio_transformer_exclusions():
    """
    Test de exclusión de columnas.
    """
    print("\n" + "="*60)
    print("Test de exclusión de columnas")
    print("="*60)
    
    data = []
    
    for i, month in enumerate(range(202101, 202105)):
        data.append({
            'numero_de_cliente': 1,
            'foto_mes': month,
            'mcuentas_saldo': 100 + 10 * i,
            'cliente_edad': 30,  # Debe ser excluida
            'mprestamos_total_lag1': 50,  # Debe ser excluida (contiene _lag)
            'mprestamos_delta': 10,  # Debe ser excluida (contiene _delta)
        })
    
    df = pd.DataFrame(data)
    
    transformer = AvgRatioTransformer(months=2)
    transformer.fit(df)
    transformed_df = transformer.transform(df)
    
    new_cols = [col for col in transformed_df.columns if col not in df.columns]
    
    print(f"\nColumnas nuevas creadas: {new_cols}")
    
    # Verificar que no se crearon columnas para las excluidas
    excluded_patterns = ['foto_mes', 'numero_de_cliente', 'cliente_edad', 'lag1', 'delta']
    
    test_pass = True
    for pattern in excluded_patterns:
        if any(pattern in col and ('_avg_' in col or '_ratio_' in col) for col in new_cols):
            print(f"❌ Se creó una columna para el patrón excluido: {pattern}")
            test_pass = False
    
    # Verificar que SÍ se creó para mcuentas_saldo
    if 'mcuentas_saldo_avg_ratio_2m' in new_cols:
        print("✅ Se creó columna de ratio para mcuentas_saldo")
    else:
        print("❌ No se creó columna de ratio para mcuentas_saldo")
        test_pass = False
    
    if test_pass:
        print("\n✅ Test de exclusiones PASS")
        return True
    else:
        print("\n❌ Test de exclusiones FAIL")
        return False


if __name__ == "__main__":
    print("Ejecutando tests del AvgRatioTransformer...\n")
    
    # Test básico
    success_basic = test_avg_ratio_transformer_basic()
    
    # Test con 3 meses
    success_3months = test_avg_ratio_transformer_months_3()
    
    # Test de exclusiones
    success_exclusions = test_avg_ratio_transformer_exclusions()
    
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"Test básico: {'✓ PASS' if success_basic else '✗ FAIL'}")
    print(f"Test con 3 meses: {'✓ PASS' if success_3months else '✗ FAIL'}")
    print(f"Test de exclusiones: {'✓ PASS' if success_exclusions else '✗ FAIL'}")
    
    if success_basic and success_3months and success_exclusions:
        print("\n🎉 TODOS LOS TESTS PASARON EXITOSAMENTE!")
        sys.exit(0)
    else:
        print("\n❌ ALGUNOS TESTS FALLARON.")
        sys.exit(1)

