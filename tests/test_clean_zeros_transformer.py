import pandas as pd
import numpy as np
import sys
import os

# Agregar el directorio src al path para importar los módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dmeyf2025.processors.feature_processors import CleanZerosTransformer


def test_clean_zeros_transformer():
    """
    Test de CleanZerosTransformer con pares de variables cVARIABLE y mVARIABLE.
    """
    print("="*60)
    print("Test de CleanZerosTransformer")
    print("="*60)
    
    # Crear datos sintéticos que imitan el formato real del dataset
    data = pd.DataFrame({
        'numero_de_cliente': [1, 2, 3, 4, 5],
        'foto_mes': [202101, 202101, 202101, 202101, 202101],
        # Par 1: ccuentas_corrientes - mcuentas_corrientes
        'ccuentas_corrientes': [0, 2, 0, 5, 1],
        'mcuentas_corrientes': [0, 1500.0, 0, 7500.0, 1000.0],
        # Par 2: cprestamos_personales - mprestamos_personales
        'cprestamos_personales': [0, 0, 3, 0, 2],
        'mprestamos_personales': [0, 0, 45000.0, 0, 30000.0],
        # Par 3: ctarjetas_visa - mtarjetas_visa
        'ctarjetas_visa': [1, 0, 0, 2, 0],
        'mtarjetas_visa': [2000.0, 0, 0, 5500.0, 0],
        # Variable sin par (solo c)
        'csolo_cantidad': [0, 1, 2, 0, 3],
        # Variable sin par (solo m)
        'msolo_monto': [100.0, 0, 200.0, 0, 300.0],
        # Otras variables
        'edad': [25, 35, 45, 55, 65],
    })
    
    print("\nDatos originales:")
    print(data)
    
    print(f"\nDimensiones: {data.shape[0]} filas, {data.shape[1]} columnas")
    
    # Crear y aplicar el transformador
    print("\n" + "="*60)
    print("Aplicando CleanZerosTransformer...")
    print("="*60)
    
    transformer = CleanZerosTransformer()
    transformer.fit(data)
    
    print(f"\nAplicando transformación...")
    
    transformed_data = transformer.transform(data)
    
    print("\nDatos transformados:")
    print(transformed_data)
    
    # Verificaciones
    print("\n" + "="*60)
    print("VERIFICACIONES")
    print("="*60)
    
    all_tests_passed = True
    
    # Test 1: Verificar que la transformación se aplicó
    print("\n--- Test 1: Transformación aplicada ---")
    test1_pass = transformed_data is not None
    if test1_pass:
        print("✅ Test 1 PASS: Transformación aplicada correctamente")
    else:
        print("❌ Test 1 FAIL: Error en transformación")
        all_tests_passed = False
    
    # Test 2: Cliente 1 - ccuentas_corrientes=0, debe poner mcuentas_corrientes en NaN
    print("\n--- Test 2: Cliente 1 (cantidad=0) ---")
    c1_mcuentas = transformed_data.loc[0, 'mcuentas_corrientes']
    print(f"mcuentas_corrientes: {c1_mcuentas} (esperado: NaN porque ccuentas_corrientes=0)")
    
    test2_pass = pd.isna(c1_mcuentas)
    if test2_pass:
        print("✅ Test 2 PASS: Monto convertido a NaN cuando cantidad=0")
    else:
        print("❌ Test 2 FAIL: Monto no convertido a NaN")
        all_tests_passed = False
    
    # Test 3: Cliente 2 - ccuentas_corrientes=2, NO debe modificar mcuentas_corrientes
    print("\n--- Test 3: Cliente 2 (cantidad>0) ---")
    c2_mcuentas = transformed_data.loc[1, 'mcuentas_corrientes']
    print(f"mcuentas_corrientes: {c2_mcuentas} (esperado: 1500.0 porque ccuentas_corrientes=2)")
    
    test3_pass = c2_mcuentas == 1500.0
    if test3_pass:
        print("✅ Test 3 PASS: Monto preservado cuando cantidad>0")
    else:
        print("❌ Test 3 FAIL: Monto modificado incorrectamente")
        all_tests_passed = False
    
    # Test 4: Cliente 3 - múltiples pares con diferentes valores
    print("\n--- Test 4: Cliente 3 (múltiples pares) ---")
    c3_mcuentas = transformed_data.loc[2, 'mcuentas_corrientes']
    c3_mprestamos = transformed_data.loc[2, 'mprestamos_personales']
    c3_mtarjetas = transformed_data.loc[2, 'mtarjetas_visa']
    
    print(f"ccuentas_corrientes=0 -> mcuentas_corrientes: {c3_mcuentas} (esperado: NaN)")
    print(f"cprestamos_personales=3 -> mprestamos_personales: {c3_mprestamos} (esperado: 45000.0)")
    print(f"ctarjetas_visa=0 -> mtarjetas_visa: {c3_mtarjetas} (esperado: NaN)")
    
    test4_pass = (pd.isna(c3_mcuentas) and 
                  c3_mprestamos == 45000.0 and 
                  pd.isna(c3_mtarjetas))
    
    if test4_pass:
        print("✅ Test 4 PASS: Múltiples pares manejados correctamente")
    else:
        print("❌ Test 4 FAIL: Error en manejo de múltiples pares")
        all_tests_passed = False
    
    # Test 5: Cliente 2 - cprestamos_personales=0 y mprestamos_personales=0
    print("\n--- Test 5: Cliente 2 (cantidad=0, monto=0) ---")
    c2_mprestamos = transformed_data.loc[1, 'mprestamos_personales']
    print(f"cprestamos_personales=0 -> mprestamos_personales: {c2_mprestamos} (esperado: NaN)")
    
    test5_pass = pd.isna(c2_mprestamos)
    if test5_pass:
        print("✅ Test 5 PASS: Monto 0 convertido a NaN cuando cantidad=0")
    else:
        print("❌ Test 5 FAIL: Monto 0 no convertido a NaN")
        all_tests_passed = False
    
    # Test 6: Variables sin par NO deben modificarse
    print("\n--- Test 6: Variables sin par ---")
    original_csolo = data['csolo_cantidad'].tolist()
    transformed_csolo = transformed_data['csolo_cantidad'].tolist()
    original_msolo = data['msolo_monto'].tolist()
    transformed_msolo = transformed_data['msolo_monto'].tolist()
    
    print(f"csolo_cantidad original: {original_csolo}")
    print(f"csolo_cantidad transformado: {transformed_csolo}")
    print(f"msolo_monto original: {original_msolo}")
    print(f"msolo_monto transformado: {transformed_msolo}")
    
    test6_pass = (original_csolo == transformed_csolo and 
                  original_msolo == transformed_msolo)
    
    if test6_pass:
        print("✅ Test 6 PASS: Variables sin par no modificadas")
    else:
        print("❌ Test 6 FAIL: Variables sin par modificadas incorrectamente")
        all_tests_passed = False
    
    # Test 7: Otras variables NO deben modificarse
    print("\n--- Test 7: Otras variables (edad, etc.) ---")
    original_edad = data['edad'].tolist()
    transformed_edad = transformed_data['edad'].tolist()
    
    test7_pass = original_edad == transformed_edad
    if test7_pass:
        print("✅ Test 7 PASS: Otras variables no modificadas")
    else:
        print("❌ Test 7 FAIL: Otras variables modificadas")
        all_tests_passed = False
    
    # Test 8: Conteo de valores modificados
    print("\n--- Test 8: Conteo de valores convertidos a NaN ---")
    
    # Pares conocidos: ccuentas_corrientes <-> mcuentas_corrientes, 
    #                  cprestamos_personales <-> mprestamos_personales,
    #                  ctarjetas_visa <-> mtarjetas_visa
    pares_conocidos = [
        ('ccuentas_corrientes', 'mcuentas_corrientes'),
        ('cprestamos_personales', 'mprestamos_personales'),
        ('ctarjetas_visa', 'mtarjetas_visa')
    ]
    
    for c_col, m_col in pares_conocidos:
        original_nans = data[m_col].isna().sum()
        transformed_nans = transformed_data[m_col].isna().sum()
        zeros_in_c = (data[c_col] == 0).sum()
        expected_new_nans = zeros_in_c
        actual_new_nans = transformed_nans - original_nans
        
        print(f"\n{c_col} <-> {m_col}:")
        print(f"  Cantidad=0: {zeros_in_c} casos")
        print(f"  NaNs originales en monto: {original_nans}")
        print(f"  NaNs nuevos en monto: {actual_new_nans}")
        print(f"  NaNs totales en monto: {transformed_nans}")
    
    # Resumen visual
    print("\n" + "="*60)
    print("RESUMEN VISUAL")
    print("="*60)
    
    comparison_cols = ['numero_de_cliente', 'ccuentas_corrientes', 'mcuentas_corrientes',
                       'cprestamos_personales', 'mprestamos_personales', 
                       'ctarjetas_visa', 'mtarjetas_visa']
    
    print("\nDatos originales (subset):")
    print(data[comparison_cols])
    
    print("\nDatos transformados (subset):")
    print(transformed_data[comparison_cols])
    
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


def test_clean_zeros_edge_cases():
    """
    Test de casos especiales.
    """
    print("\n" + "="*60)
    print("Test de casos especiales")
    print("="*60)
    
    all_pass = True
    
    # Caso 1: Dataset sin pares de variables
    print("\n--- Caso 1: Dataset sin pares ---")
    df_no_pairs = pd.DataFrame({
        'numero_de_cliente': [1, 2, 3],
        'foto_mes': [202101, 202101, 202101],
        'cvariable1': [0, 1, 2],
        'variable2': [100, 200, 300],
        'mvariable3': [50, 0, 75],
    })
    
    transformer = CleanZerosTransformer()
    transformer.fit(df_no_pairs)
    result = transformer.transform(df_no_pairs)
    
    print(f"Transformación aplicada sin pares")
    # Si no hay pares, el DataFrame permanece igual (excepto limpieza de 202006/tmobile_app)
    test1_pass = result is not None
    
    if test1_pass:
        print("✅ Caso 1 PASS: Transformación aplicada correctamente sin pares")
    else:
        print("❌ Caso 1 FAIL")
        all_pass = False
    
    # Caso 2: Todos los valores de cantidad son cero
    print("\n--- Caso 2: Todas las cantidades en cero ---")
    df_all_zeros = pd.DataFrame({
        'numero_de_cliente': [1, 2, 3],
        'foto_mes': [202101, 202101, 202101],
        'cprestamos': [0, 0, 0],
        'mprestamos': [1000.0, 2000.0, 3000.0],
    })
    
    transformer = CleanZerosTransformer()
    transformer.fit(df_all_zeros)
    result = transformer.transform(df_all_zeros)
    
    all_nans = result['mprestamos'].isna().all()
    print(f"Todos los montos convertidos a NaN: {all_nans}")
    
    if all_nans:
        print("✅ Caso 2 PASS: Todos los montos convertidos a NaN")
    else:
        print("❌ Caso 2 FAIL")
        all_pass = False
    
    # Caso 3: Ningún valor de cantidad es cero
    print("\n--- Caso 3: Ninguna cantidad en cero ---")
    df_no_zeros = pd.DataFrame({
        'numero_de_cliente': [1, 2, 3],
        'foto_mes': [202101, 202101, 202101],
        'ctarjetas': [1, 2, 3],
        'mtarjetas': [1000.0, 2000.0, 3000.0],
    })
    
    transformer = CleanZerosTransformer()
    transformer.fit(df_no_zeros)
    result = transformer.transform(df_no_zeros)
    
    no_nans = result['mtarjetas'].notna().all()
    print(f"Ningún monto convertido a NaN: {no_nans}")
    
    if no_nans:
        print("✅ Caso 3 PASS: Ningún monto modificado")
    else:
        print("❌ Caso 3 FAIL")
        all_pass = False
    
    # Caso 4: Montos ya tienen algunos NaN
    print("\n--- Caso 4: Montos con NaN preexistentes ---")
    df_with_nans = pd.DataFrame({
        'numero_de_cliente': [1, 2, 3, 4],
        'foto_mes': [202101, 202101, 202101, 202101],
        'ccheques': [0, 1, 0, 2],
        'mcheques': [100.0, np.nan, 300.0, 400.0],
    })
    
    transformer = CleanZerosTransformer()
    transformer.fit(df_with_nans)
    result = transformer.transform(df_with_nans)
    
    print(f"mcheques original: {df_with_nans['mcheques'].tolist()}")
    print(f"mcheques transformado: {result['mcheques'].tolist()}")
    
    # Cliente 1: cantidad=0, debe ser NaN
    # Cliente 2: cantidad=1, ya era NaN, debe seguir NaN
    # Cliente 3: cantidad=0, debe ser NaN
    # Cliente 4: cantidad=2, debe seguir 400.0
    
    test4_pass = (pd.isna(result.loc[0, 'mcheques']) and
                  pd.isna(result.loc[1, 'mcheques']) and
                  pd.isna(result.loc[2, 'mcheques']) and
                  result.loc[3, 'mcheques'] == 400.0)
    
    if test4_pass:
        print("✅ Caso 4 PASS: Maneja NaN preexistentes correctamente")
    else:
        print("❌ Caso 4 FAIL")
        all_pass = False
    
    if all_pass:
        print("\n✅ Todos los casos especiales pasaron")
        return True
    else:
        print("\n❌ Algunos casos especiales fallaron")
        return False


if __name__ == "__main__":
    print("Ejecutando tests del CleanZerosTransformer...\n")
    
    # Test principal
    success_main = test_clean_zeros_transformer()
    
    # Test de casos especiales
    success_edge_cases = test_clean_zeros_edge_cases()
    
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"Test principal: {'✓ PASS' if success_main else '✗ FAIL'}")
    print(f"Test de casos especiales: {'✓ PASS' if success_edge_cases else '✗ FAIL'}")
    
    if success_main and success_edge_cases:
        print("\n🎉 TODOS LOS TESTS PASARON EXITOSAMENTE!")
        sys.exit(0)
    else:
        print("\n❌ ALGUNOS TESTS FALLARON.")
        sys.exit(1)

