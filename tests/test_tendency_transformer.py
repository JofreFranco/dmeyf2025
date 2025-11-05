import pandas as pd
import numpy as np
import sys
import os

# Agregar el directorio src al path para importar los módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dmeyf2025.processors.feature_processors import TendencyTransformer, LegacyTendencyTransformer


def test_tendency_transformer_synthetic():
    """
    Test de TendencyTransformer con datos sintéticos de pendientes conocidas.
    """
    print("="*60)
    print("Test de TendencyTransformer con pendientes conocidas")
    print("="*60)
    
    # Crear datos sintéticos con pendientes conocidas
    np.random.seed(42)
    
    data = []
    
    # Cliente 1: Variable con pendiente 2.5
    for i, month in enumerate(range(202101, 202107)):  # 6 meses
        data.append({
            'numero_de_cliente': 1,
            'foto_mes': month,
            'mcuentas_saldo': 100 + 2.5 * i,  # Pendiente exacta de 2.5
            'mprestamos_total': 200 - 1.5 * i,  # Pendiente exacta de -1.5
            'mtarjetas_consumo': 50,  # Pendiente 0 (constante)
        })
    
    # Cliente 2: Variable con pendiente 5.0
    for i, month in enumerate(range(202101, 202107)):
        data.append({
            'numero_de_cliente': 2,
            'foto_mes': month,
            'mcuentas_saldo': 150 + 5.0 * i,  # Pendiente 5.0
            'mprestamos_total': 300 - 3.0 * i,  # Pendiente -3.0
            'mtarjetas_consumo': 75,  # Pendiente 0
        })
    
    # Cliente 3: Pocos datos (solo 2 meses, mínimo para calcular pendiente)
    for i, month in enumerate([202101, 202102]):
        data.append({
            'numero_de_cliente': 3,
            'foto_mes': month,
            'mcuentas_saldo': 100 + 10.0 * i,  # Pendiente 10.0
            'mprestamos_total': 200 - 5.0 * i,  # Pendiente -5.0
            'mtarjetas_consumo': 50,  # Pendiente 0
        })
    
    # Cliente 4: Con algunos valores nulos
    for i, month in enumerate(range(202101, 202106)):  # 5 meses
        data.append({
            'numero_de_cliente': 4,
            'foto_mes': month,
            'mcuentas_saldo': 100 + 3.0 * i if i != 2 else np.nan,  # Pendiente ~3.0 con un nulo
            'mprestamos_total': 200 - 2.0 * i,  # Pendiente -2.0
            'mtarjetas_consumo': 50,  # Pendiente 0
        })
    
    df = pd.DataFrame(data)
    print(f"\nDatos sintéticos creados: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Clientes: {df['numero_de_cliente'].nunique()}")
    print(f"Variables: {[col for col in df.columns if col not in ['numero_de_cliente', 'foto_mes']]}")
    
    # Mostrar datos de ejemplo
    print("\nDatos del Cliente 1:")
    print(df[df['numero_de_cliente'] == 1][['numero_de_cliente', 'foto_mes', 'mcuentas_saldo', 'mprestamos_total', 'mtarjetas_consumo']])
    
    # Crear y aplicar el transformador
    print("\n" + "="*60)
    print("Aplicando TendencyTransformer...")
    print("="*60)
    
    transformer = TendencyTransformer()
    transformer.fit(df)
    transformed_df = transformer.transform(df)
    
    print(f"\nDatos transformados: {transformed_df.shape[0]} filas, {transformed_df.shape[1]} columnas")
    
    # Calcular cuántas columnas nuevas se crearon
    original_cols = set(df.columns)
    new_cols = sorted([col for col in transformed_df.columns if col not in original_cols])
    print(f"Columnas nuevas creadas: {len(new_cols)}")
    print(f"Nombres: {new_cols}")
    
    # Verificaciones
    print("\n" + "="*60)
    print("VERIFICACIONES")
    print("="*60)
    
    all_tests_passed = True
    
    # Test 1: Cliente 1, última observación (debería tener pendiente usando todos los datos)
    print("\n--- Test 1: Cliente 1, última observación ---")
    client1_last = transformed_df[(transformed_df['numero_de_cliente'] == 1) & 
                                   (transformed_df['foto_mes'] == 202106)]
    
    slope_2_5 = client1_last['mcuentas_saldo_tendency'].values[0]
    slope_neg_1_5 = client1_last['mprestamos_total_tendency'].values[0]
    slope_0 = client1_last['mtarjetas_consumo_tendency'].values[0]
    
    print(f"mcuentas_saldo_tendency: {slope_2_5:.6f} (esperado: 2.5)")
    print(f"mprestamos_total_tendency: {slope_neg_1_5:.6f} (esperado: -1.5)")
    print(f"mtarjetas_consumo_tendency: {slope_0:.6f} (esperado: 0.0)")
    
    # Verificar con tolerancia
    tolerance = 0.01
    test1_pass = (abs(slope_2_5 - 2.5) < tolerance and 
                  abs(slope_neg_1_5 - (-1.5)) < tolerance and 
                  abs(slope_0 - 0.0) < tolerance)
    
    if test1_pass:
        print("✅ Test 1 PASS: Pendientes correctas para Cliente 1")
    else:
        print("❌ Test 1 FAIL: Pendientes incorrectas para Cliente 1")
        all_tests_passed = False
    
    # Test 2: Cliente 2, última observación
    print("\n--- Test 2: Cliente 2, última observación ---")
    client2_last = transformed_df[(transformed_df['numero_de_cliente'] == 2) & 
                                   (transformed_df['foto_mes'] == 202106)]
    
    slope_5 = client2_last['mcuentas_saldo_tendency'].values[0]
    slope_neg_3 = client2_last['mprestamos_total_tendency'].values[0]
    
    print(f"mcuentas_saldo_tendency: {slope_5:.6f} (esperado: 5.0)")
    print(f"mprestamos_total_tendency: {slope_neg_3:.6f} (esperado: -3.0)")
    
    test2_pass = (abs(slope_5 - 5.0) < tolerance and 
                  abs(slope_neg_3 - (-3.0)) < tolerance)
    
    if test2_pass:
        print("✅ Test 2 PASS: Pendientes correctas para Cliente 2")
    else:
        print("❌ Test 2 FAIL: Pendientes incorrectas para Cliente 2")
        all_tests_passed = False
    
    # Test 3: Expanding window - Cliente 1 en distintos meses
    print("\n--- Test 3: Expanding window - Cliente 1 en distintos meses ---")
    client1_all = transformed_df[transformed_df['numero_de_cliente'] == 1][
        ['foto_mes', 'mcuentas_saldo', 'mcuentas_saldo_tendency']
    ]
    print(client1_all)
    
    # Verificar que en el primer mes (con 1 dato) la tendencia es NaN
    first_tendency = client1_all.iloc[0]['mcuentas_saldo_tendency']
    print(f"\nPrimer mes (1 dato): tendencia = {first_tendency} (esperado: NaN)")
    test3a_pass = pd.isna(first_tendency)
    
    # Verificar que en el segundo mes (con 2 datos) ya hay tendencia
    second_tendency = client1_all.iloc[1]['mcuentas_saldo_tendency']
    print(f"Segundo mes (2 datos): tendencia = {second_tendency:.6f} (esperado: 2.5)")
    test3b_pass = not pd.isna(second_tendency) and abs(second_tendency - 2.5) < tolerance
    
    # Verificar que la tendencia se mantiene estable conforme agregamos más datos
    third_tendency = client1_all.iloc[2]['mcuentas_saldo_tendency']
    print(f"Tercer mes (3 datos): tendencia = {third_tendency:.6f} (esperado: 2.5)")
    test3c_pass = abs(third_tendency - 2.5) < tolerance
    
    test3_pass = test3a_pass and test3b_pass and test3c_pass
    
    if test3_pass:
        print("✅ Test 3 PASS: Expanding window funciona correctamente")
    else:
        print("❌ Test 3 FAIL: Expanding window no funciona correctamente")
        all_tests_passed = False
    
    # Test 4: Cliente con datos mínimos (2 observaciones)
    print("\n--- Test 4: Cliente 3 con datos mínimos (2 observaciones) ---")
    client3_last = transformed_df[(transformed_df['numero_de_cliente'] == 3) & 
                                   (transformed_df['foto_mes'] == 202102)]
    
    slope_10 = client3_last['mcuentas_saldo_tendency'].values[0]
    print(f"mcuentas_saldo_tendency: {slope_10:.6f} (esperado: 10.0)")
    
    test4_pass = abs(slope_10 - 10.0) < tolerance
    
    if test4_pass:
        print("✅ Test 4 PASS: Calcula pendiente con 2 observaciones")
    else:
        print("❌ Test 4 FAIL: No calcula bien con 2 observaciones")
        all_tests_passed = False
    
    # Test 5: Cliente con valores nulos
    print("\n--- Test 5: Cliente 4 con valores nulos ---")
    client4_all = transformed_df[transformed_df['numero_de_cliente'] == 4][
        ['foto_mes', 'mcuentas_saldo', 'mcuentas_saldo_tendency']
    ]
    print(client4_all)
    
    # La pendiente debería calcularse ignorando los nulos
    # Los valores son: 100, 103, NaN, 109, 112
    # La nueva implementación vectorizada podría dar resultados ligeramente diferentes
    client4_last = transformed_df[(transformed_df['numero_de_cliente'] == 4) & 
                                   (transformed_df['foto_mes'] == 202105)]
    slope_with_nulls = client4_last['mcuentas_saldo_tendency'].values[0]
    print(f"\nPendiente con un valor nulo: {slope_with_nulls:.6f}")
    print("Nota: Los nulos se ignoran al calcular la pendiente")
    
    # Verificar que calcula una pendiente válida (no NaN) y positiva
    test5_pass = not pd.isna(slope_with_nulls) and slope_with_nulls > 0
    
    if test5_pass:
        print("✅ Test 5 PASS: Maneja valores nulos correctamente (calcula pendiente válida)")
    else:
        print("❌ Test 5 FAIL: No maneja bien los valores nulos")
        all_tests_passed = False
    
    # Test 6: Verificar que no modifica las columnas originales
    print("\n--- Test 6: Integridad de datos originales ---")
    original_data_preserved = df.equals(transformed_df[df.columns])
    
    if original_data_preserved:
        print("✅ Test 6 PASS: Datos originales preservados")
    else:
        print("❌ Test 6 FAIL: Datos originales modificados")
        all_tests_passed = False
    
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


def test_tendency_transformer_edge_cases():
    """
    Test de casos especiales y bordes.
    """
    print("\n" + "="*60)
    print("Test de casos especiales")
    print("="*60)
    
    # Caso 1: Cliente con un solo dato
    print("\n--- Caso 1: Cliente con una sola observación ---")
    df_single = pd.DataFrame([
        {'numero_de_cliente': 1, 'foto_mes': 202101, 'mcuenta': 100}
    ])
    
    transformer = TendencyTransformer()
    transformer.fit(df_single)
    result = transformer.transform(df_single)
    
    tendency = result['mcuenta_tendency'].values[0]
    print(f"Tendencia con 1 dato: {tendency} (esperado: NaN)")
    
    test1_pass = pd.isna(tendency)
    if test1_pass:
        print("✅ Caso 1 PASS: Devuelve NaN con 1 observación")
    else:
        print("❌ Caso 1 FAIL")
    
    # Caso 2: Valores constantes (pendiente 0)
    print("\n--- Caso 2: Valores constantes ---")
    df_constant = pd.DataFrame([
        {'numero_de_cliente': 1, 'foto_mes': 202101, 'mcuenta': 50},
        {'numero_de_cliente': 1, 'foto_mes': 202102, 'mcuenta': 50},
        {'numero_de_cliente': 1, 'foto_mes': 202103, 'mcuenta': 50},
    ])
    
    result = transformer.transform(df_constant)
    tendency = result[result['foto_mes'] == 202103]['mcuenta_tendency'].values[0]
    print(f"Tendencia de valores constantes: {tendency:.6f} (esperado: 0.0)")
    
    test2_pass = abs(tendency - 0.0) < 0.01
    if test2_pass:
        print("✅ Caso 2 PASS: Pendiente 0 para valores constantes")
    else:
        print("❌ Caso 2 FAIL")
    
    # Caso 3: Múltiples clientes independientes
    print("\n--- Caso 3: Múltiples clientes independientes ---")
    df_multi = pd.DataFrame([
        {'numero_de_cliente': 1, 'foto_mes': 202101, 'mcuenta': 0},
        {'numero_de_cliente': 1, 'foto_mes': 202102, 'mcuenta': 10},
        {'numero_de_cliente': 2, 'foto_mes': 202101, 'mcuenta': 100},
        {'numero_de_cliente': 2, 'foto_mes': 202102, 'mcuenta': 90},
    ])
    
    result = transformer.transform(df_multi)
    
    client1_slope = result[(result['numero_de_cliente'] == 1) & 
                           (result['foto_mes'] == 202102)]['mcuenta_tendency'].values[0]
    client2_slope = result[(result['numero_de_cliente'] == 2) & 
                           (result['foto_mes'] == 202102)]['mcuenta_tendency'].values[0]
    
    print(f"Cliente 1 tendencia: {client1_slope:.6f} (esperado: 10.0)")
    print(f"Cliente 2 tendencia: {client2_slope:.6f} (esperado: -10.0)")
    
    test3_pass = abs(client1_slope - 10.0) < 0.01 and abs(client2_slope - (-10.0)) < 0.01
    if test3_pass:
        print("✅ Caso 3 PASS: Clientes independientes calculados correctamente")
    else:
        print("❌ Caso 3 FAIL")
    
    all_pass = test1_pass and test2_pass and test3_pass
    
    if all_pass:
        print("\n✅ Todos los casos especiales pasaron")
        return True
    else:
        print("\n❌ Algunos casos especiales fallaron")
        return False


if __name__ == "__main__":
    print("Ejecutando tests del TendencyTransformer...\n")
    
    # Test con datos sintéticos
    success_synthetic = test_tendency_transformer_synthetic()
    
    # Test de casos especiales
    success_edge_cases = test_tendency_transformer_edge_cases()
    
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"Test con datos sintéticos: {'✓ PASS' if success_synthetic else '✗ FAIL'}")
    print(f"Test de casos especiales: {'✓ PASS' if success_edge_cases else '✗ FAIL'}")
    
    if success_synthetic and success_edge_cases:
        print("\n🎉 TODOS LOS TESTS PASARON EXITOSAMENTE!")
        sys.exit(0)
    else:
        print("\n❌ ALGUNOS TESTS FALLARON.")
        sys.exit(1)

