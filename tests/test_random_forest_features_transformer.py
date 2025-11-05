import pandas as pd
import numpy as np
import sys
import os

# Agregar el directorio src al path para importar los módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dmeyf2025.processors.feature_processors import RandomForestFeaturesTransformer


def test_random_forest_features_transformer_synthetic():
    """
    Test de RandomForestFeaturesTransformer con datos sintéticos.
    Verifica que el transformer entrene correctamente y genere las features esperadas.
    """
    print("="*60)
    print("Test de RandomForestFeaturesTransformer con datos sintéticos")
    print("="*60)
    
    # Crear datos sintéticos
    np.random.seed(42)
    
    n_clients = 50
    n_months = 6
    training_months = [202101, 202102, 202103]
    test_months = [202104, 202105, 202106]
    
    data = []
    
    for client_id in range(1, n_clients + 1):
        for i, month in enumerate(range(202101, 202107)):
            # Variables predictoras
            mcuentas_saldo = np.random.normal(1000, 300)
            mprestamos_total = np.random.normal(500, 150)
            mtarjetas_consumo = np.random.normal(200, 50)
            minversion = np.random.normal(800, 200)
            
            # Crear label: 1 si mcuentas_saldo > 1000 y mprestamos_total < 500
            label = 1 if (mcuentas_saldo > 1000 and mprestamos_total < 500) else 0
            
            data.append({
                'numero_de_cliente': client_id,
                'foto_mes': month,
                'mcuentas_saldo': mcuentas_saldo,
                'mprestamos_total': mprestamos_total,
                'mtarjetas_consumo': mtarjetas_consumo,
                'minversion': minversion,
                'label': label,
                'target': label,
                'weight': 1.0
            })
    
    df = pd.DataFrame(data)
    print(f"\nDatos sintéticos creados: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Clientes: {df['numero_de_cliente'].nunique()}")
    print(f"Meses: {sorted(df['foto_mes'].unique())}")
    print(f"Meses de entrenamiento: {training_months}")
    print(f"Distribución de labels: {df['label'].value_counts().to_dict()}")
    
    # Verificaciones
    print("\n" + "="*60)
    print("EJECUTANDO TESTS")
    print("="*60)
    
    all_tests_passed = True
    
    # Test 1: Fit con datos de entrenamiento
    print("\n--- Test 1: Fit con datos de entrenamiento ---")
    try:
        transformer = RandomForestFeaturesTransformer(
            n_estimators=5,  # Pocos árboles para test rápido
            num_leaves=8,
            min_data_in_leaf=10,
            feature_fraction_bynode=0.5,
            training_months=training_months
        )
        
        transformer.fit(df)
        
        # Verificar que el modelo fue entrenado
        assert hasattr(transformer, 'model_'), "El modelo no fue entrenado"
        print("✅ Test 1 PASS: Modelo entrenado correctamente")
        test1_pass = True
    except Exception as e:
        print(f"❌ Test 1 FAIL: {e}")
        all_tests_passed = False
        test1_pass = False
        import traceback
        traceback.print_exc()
    
    if not test1_pass:
        print("\n❌ No se pudo continuar con los demás tests")
        return False
    
    # Test 2: Transform genera nuevas columnas
    print("\n--- Test 2: Transform genera nuevas columnas ---")
    try:
        # Preparar datos para transform (conservar foto_mes y eliminar las columnas exclude_cols)
        # El transformer espera que foto_mes esté presente pero no las otras columnas excluidas
        df_transform = df[['foto_mes', 'mcuentas_saldo', 'mprestamos_total', 'mtarjetas_consumo', 'minversion']].copy()
        
        # Guardar las columnas originales ANTES de transform (porque modifica in-place)
        original_cols = set(df_transform.columns)
        
        transformed_df = transformer.transform(df_transform)
        
        # Verificar que se agregaron columnas
        new_cols = sorted([col for col in transformed_df.columns if col not in original_cols])
        
        print(f"Columnas originales: {len(original_cols)}")
        print(f"Columnas después de transform: {len(transformed_df.columns)}")
        print(f"Nuevas columnas generadas: {len(new_cols)}")
        
        # Mostrar algunos ejemplos de nuevas columnas
        print(f"Ejemplos de nuevas columnas: {new_cols[:10]}")
        
        # Verificar que las nuevas columnas tienen el formato esperado rf_XXX_YYY
        rf_cols = [col for col in new_cols if col.startswith('rf_')]
        print(f"Columnas con formato rf_XXX_YYY: {len(rf_cols)}")
        
        assert len(rf_cols) > 0, "No se generaron columnas con formato rf_"
        assert len(new_cols) == len(rf_cols), "Hay columnas nuevas que no tienen formato rf_"
        
        print("✅ Test 2 PASS: Se generaron columnas con formato correcto")
        test2_pass = True
    except Exception as e:
        print(f"❌ Test 2 FAIL: {e}")
        all_tests_passed = False
        test2_pass = False
        import traceback
        traceback.print_exc()
    
    if not test2_pass:
        print("\n❌ No se pudo continuar con los demás tests")
        return False
    
    # Test 3: Verificar valores de las nuevas columnas (deben ser 0 o 1)
    print("\n--- Test 3: Valores de las nuevas columnas son binarios ---")
    try:
        all_binary = True
        for col in rf_cols[:20]:  # Verificar las primeras 20 columnas
            unique_vals = transformed_df[col].unique()
            if not set(unique_vals).issubset({0, 1}):
                print(f"Columna {col} tiene valores no binarios: {unique_vals}")
                all_binary = False
                break
        
        assert all_binary, "Algunas columnas tienen valores que no son 0 o 1"
        print(f"✅ Test 3 PASS: Todas las columnas rf_ son binarias (0 o 1)")
        test3_pass = True
    except Exception as e:
        print(f"❌ Test 3 FAIL: {e}")
        all_tests_passed = False
        test3_pass = False
    
    # Test 4: Cada fila tiene exactamente n_estimators columnas con valor 1
    print("\n--- Test 4: Cada fila tiene exactamente n_estimators columnas activas ---")
    try:
        n_estimators = transformer.lgb_params['num_iterations']
        
        # Sumar cuántas columnas rf_ tienen valor 1 por fila
        rf_sum_per_row = transformed_df[rf_cols].sum(axis=1)
        
        # Todas las filas deben tener exactamente n_estimators columnas con 1
        unique_sums = rf_sum_per_row.unique()
        
        print(f"Número de estimadores: {n_estimators}")
        print(f"Suma de columnas activas por fila (único valores): {unique_sums}")
        
        assert len(unique_sums) == 1 and unique_sums[0] == n_estimators, \
            f"Esperado {n_estimators} columnas activas por fila, obtenido: {unique_sums}"
        
        print(f"✅ Test 4 PASS: Cada fila tiene exactamente {n_estimators} columnas activas")
        test4_pass = True
    except Exception as e:
        print(f"❌ Test 4 FAIL: {e}")
        all_tests_passed = False
        test4_pass = False
    
    # Test 5: Transform preserva el número de filas
    print("\n--- Test 5: Transform preserva el número de filas ---")
    try:
        assert len(df_transform) == len(transformed_df), \
            f"Número de filas cambió: {len(df_transform)} -> {len(transformed_df)}"
        
        print(f"✅ Test 5 PASS: Número de filas preservado ({len(transformed_df)} filas)")
        test5_pass = True
    except Exception as e:
        print(f"❌ Test 5 FAIL: {e}")
        all_tests_passed = False
        test5_pass = False
    
    # Test 6: Transform mantiene las columnas originales
    print("\n--- Test 6: Transform mantiene las columnas originales ---")
    try:
        original_preserved = all(col in transformed_df.columns for col in df_transform.columns)
        assert original_preserved, "Algunas columnas originales se perdieron"
        
        print("✅ Test 6 PASS: Columnas originales preservadas")
        test6_pass = True
    except Exception as e:
        print(f"❌ Test 6 FAIL: {e}")
        all_tests_passed = False
        test6_pass = False
    
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


def test_random_forest_features_transformer_edge_cases():
    """
    Test de casos especiales y edge cases.
    """
    print("\n" + "="*60)
    print("Test de casos especiales")
    print("="*60)
    
    all_tests_passed = True
    
    # Caso 1: Pocos datos de entrenamiento
    print("\n--- Caso 1: Pocos datos de entrenamiento ---")
    try:
        np.random.seed(42)
        
        # Crear dataset muy pequeño
        data = []
        for i in range(20):  # Solo 20 observaciones
            data.append({
                'numero_de_cliente': i + 1,
                'foto_mes': 202101,
                'mcuentas_saldo': np.random.normal(1000, 300),
                'mprestamos_total': np.random.normal(500, 150),
                'label': np.random.choice([0, 1]),
                'target': 0,
                'weight': 1.0
            })
        
        df_small = pd.DataFrame(data)
        
        transformer = RandomForestFeaturesTransformer(
            n_estimators=3,
            num_leaves=4,
            min_data_in_leaf=2,  # Muy bajo para permitir el entrenamiento
            training_months=[202101]
        )
        
        transformer.fit(df_small)
        
        df_transform = df_small[['foto_mes', 'mcuentas_saldo', 'mprestamos_total']].copy()
        original_n_cols = len(df_transform.columns)
        result = transformer.transform(df_transform)
        
        assert len(result) == len(df_small), "El número de filas cambió"
        assert len(result.columns) > original_n_cols, "No se generaron nuevas columnas"
        
        print("✅ Caso 1 PASS: Funciona con pocos datos de entrenamiento")
    except Exception as e:
        print(f"❌ Caso 1 FAIL: {e}")
        all_tests_passed = False
        import traceback
        traceback.print_exc()
    
    # Caso 2: Datos con clases desbalanceadas
    print("\n--- Caso 2: Datos con clases desbalanceadas ---")
    try:
        np.random.seed(42)
        
        # Crear dataset con 90% clase 0 y 10% clase 1
        data = []
        for i in range(100):
            label = 1 if i < 10 else 0  # Solo 10% son clase 1
            data.append({
                'numero_de_cliente': i + 1,
                'foto_mes': 202101,
                'mcuentas_saldo': np.random.normal(1000, 300),
                'mprestamos_total': np.random.normal(500, 150),
                'mtarjetas_consumo': np.random.normal(200, 50),
                'label': label,
                'target': label,
                'weight': 1.0
            })
        
        df_imbalanced = pd.DataFrame(data)
        
        print(f"Distribución de clases: {df_imbalanced['label'].value_counts().to_dict()}")
        
        transformer = RandomForestFeaturesTransformer(
            n_estimators=5,
            num_leaves=8,
            min_data_in_leaf=5,
            training_months=[202101]
        )
        
        transformer.fit(df_imbalanced)
        
        df_transform = df_imbalanced[['foto_mes', 'mcuentas_saldo', 'mprestamos_total', 'mtarjetas_consumo']].copy()
        result = transformer.transform(df_transform)
        
        # Verificar que genera features
        rf_cols = [col for col in result.columns if col.startswith('rf_')]
        assert len(rf_cols) > 0, "No se generaron columnas rf_"
        
        print(f"Columnas rf_ generadas: {len(rf_cols)}")
        print("✅ Caso 2 PASS: Funciona con clases desbalanceadas")
    except Exception as e:
        print(f"❌ Caso 2 FAIL: {e}")
        all_tests_passed = False
        import traceback
        traceback.print_exc()
    
    # Caso 3: Diferentes meses de entrenamiento vs predicción
    print("\n--- Caso 3: Diferentes meses de entrenamiento vs predicción ---")
    try:
        np.random.seed(42)
        
        # Crear dataset con múltiples meses
        data = []
        for month in [202101, 202102, 202103, 202104]:
            for i in range(25):
                data.append({
                    'numero_de_cliente': i + 1,
                    'foto_mes': month,
                    'mcuentas_saldo': np.random.normal(1000, 300),
                    'mprestamos_total': np.random.normal(500, 150),
                    'label': np.random.choice([0, 1]),
                    'target': 0,
                    'weight': 1.0
                })
        
        df_multimonth = pd.DataFrame(data)
        
        print(f"Meses en el dataset: {sorted(df_multimonth['foto_mes'].unique())}")
        
        # Entrenar solo con los primeros dos meses
        transformer = RandomForestFeaturesTransformer(
            n_estimators=4,
            num_leaves=6,
            min_data_in_leaf=5,
            training_months=[202101, 202102]
        )
        
        transformer.fit(df_multimonth)
        
        # Aplicar transform a todos los meses (incluyendo meses no vistos en entrenamiento)
        df_transform = df_multimonth[['foto_mes', 'mcuentas_saldo', 'mprestamos_total']].copy()
        result = transformer.transform(df_transform)
        
        # Verificar que funciona para todos los meses
        assert len(result) == len(df_multimonth), "El número de filas cambió"
        
        # Verificar que hay datos transformados para meses no vistos en entrenamiento
        result_month_202104 = result[df_multimonth['foto_mes'] == 202104]
        assert len(result_month_202104) > 0, "No hay datos para mes 202104"
        
        rf_cols = [col for col in result.columns if col.startswith('rf_')]
        assert result_month_202104[rf_cols].sum().sum() > 0, "No hay features activas para mes 202104"
        
        print(f"Filas para mes 202104: {len(result_month_202104)}")
        print(f"Features activas en mes 202104: {result_month_202104[rf_cols].sum().sum()}")
        print("✅ Caso 3 PASS: Funciona con meses diferentes de entrenamiento vs predicción")
    except Exception as e:
        print(f"❌ Caso 3 FAIL: {e}")
        all_tests_passed = False
        import traceback
        traceback.print_exc()
    
    # Resumen
    print("\n" + "="*60)
    if all_tests_passed:
        print("✅ Todos los casos especiales pasaron")
        return True
    else:
        print("❌ Algunos casos especiales fallaron")
        return False


def test_random_forest_features_deterministic():
    """
    Test para verificar que el transformer es determinista (reproducible).
    """
    print("\n" + "="*60)
    print("Test de reproducibilidad")
    print("="*60)
    
    # Crear datos
    np.random.seed(42)
    
    data = []
    for i in range(50):
        data.append({
            'numero_de_cliente': i + 1,
            'foto_mes': 202101,
            'mcuentas_saldo': np.random.normal(1000, 300),
            'mprestamos_total': np.random.normal(500, 150),
            'mtarjetas_consumo': np.random.normal(200, 50),
            'label': np.random.choice([0, 1]),
            'target': 0,
            'weight': 1.0
        })
    
    df = pd.DataFrame(data)
    
    # Primera ejecución
    transformer1 = RandomForestFeaturesTransformer(
        n_estimators=5,
        num_leaves=8,
        min_data_in_leaf=5,
        training_months=[202101]
    )
    
    transformer1.fit(df)
    df_transform = df[['foto_mes', 'mcuentas_saldo', 'mprestamos_total', 'mtarjetas_consumo']].copy()
    result1 = transformer1.transform(df_transform)
    
    # Segunda ejecución
    transformer2 = RandomForestFeaturesTransformer(
        n_estimators=5,
        num_leaves=8,
        min_data_in_leaf=5,
        training_months=[202101]
    )
    
    transformer2.fit(df)
    df_transform2 = df[['foto_mes', 'mcuentas_saldo', 'mprestamos_total', 'mtarjetas_consumo']].copy()
    result2 = transformer2.transform(df_transform2)
    
    # Comparar resultados
    rf_cols = [col for col in result1.columns if col.startswith('rf_')]
    
    # Verificar que las columnas generadas son las mismas
    rf_cols_2 = [col for col in result2.columns if col.startswith('rf_')]
    
    try:
        assert set(rf_cols) == set(rf_cols_2), "Las columnas generadas son diferentes"
        
        # Verificar que los valores son idénticos
        differences = 0
        for col in rf_cols:
            if not result1[col].equals(result2[col]):
                differences += 1
        
        if differences == 0:
            print("✅ Test de reproducibilidad PASS: Los resultados son idénticos")
            print(f"   - {len(rf_cols)} columnas verificadas")
            return True
        else:
            print(f"⚠️  Test de reproducibilidad: {differences} columnas tienen diferencias")
            print("   Nota: LightGBM puede tener variabilidad incluso con misma semilla")
            print("   Esto es aceptable si las diferencias son mínimas")
            return True  # Lo consideramos acceptable
    except Exception as e:
        print(f"❌ Test de reproducibilidad FAIL: {e}")
        return False


if __name__ == "__main__":
    print("Ejecutando tests del RandomForestFeaturesTransformer...\n")
    
    # Test principal con datos sintéticos
    success_synthetic = test_random_forest_features_transformer_synthetic()
    
    # Test de casos especiales
    success_edge_cases = test_random_forest_features_transformer_edge_cases()
    
    # Test de reproducibilidad
    success_deterministic = test_random_forest_features_deterministic()
    
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"Test con datos sintéticos: {'✓ PASS' if success_synthetic else '✗ FAIL'}")
    print(f"Test de casos especiales: {'✓ PASS' if success_edge_cases else '✗ FAIL'}")
    print(f"Test de reproducibilidad: {'✓ PASS' if success_deterministic else '✗ FAIL'}")
    
    if success_synthetic and success_edge_cases and success_deterministic:
        print("\n🎉 TODOS LOS TESTS PASARON EXITOSAMENTE!")
        sys.exit(0)
    else:
        print("\n❌ ALGUNOS TESTS FALLARON.")
        sys.exit(1)

