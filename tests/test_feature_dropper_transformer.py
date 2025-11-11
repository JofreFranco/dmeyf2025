import pandas as pd
import numpy as np
import sys
import os
import tempfile

# Agregar el directorio src al path para importar los módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dmeyf2025.processors.feature_processors import FeatureDropperTransformer


def test_feature_dropper_with_csv():
    """
    Test básico del FeatureDropperTransformer con un CSV.
    """
    print("="*60)
    print("Test de FeatureDropperTransformer con CSV")
    print("="*60)
    
    # Crear datos sintéticos
    np.random.seed(42)
    df = pd.DataFrame({
        'numero_de_cliente': range(10),
        'foto_mes': [202101] * 10,
        'mcuentas_saldo': np.random.rand(10),
        'mprestamos_total': np.random.rand(10),
        'mtarjetas_consumo': np.random.rand(10),
        'minversion': np.random.rand(10),
        'rf_001_001': np.random.randint(0, 2, 10),
        'rf_001_002': np.random.randint(0, 2, 10),
    })
    
    print(f"\nDataFrame original: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    
    # Crear un CSV temporal con features a eliminar
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
        f.write("feature\n")
        f.write("mcuentas_saldo\n")
        f.write("rf_001_001\n")
        f.write("feature_inexistente\n")  # Esta no existe en el DataFrame
    
    print(f"\nCSV temporal creado: {csv_path}")
    print("Features a eliminar: mcuentas_saldo, rf_001_001, feature_inexistente")
    
    try:
        # Test 1: Aplicar transformer con CSV
        print("\n--- Test 1: Aplicar transformer con CSV ---")
        transformer = FeatureDropperTransformer(csv_path=csv_path)
        transformer.fit(df)
        result = transformer.transform(df)
        
        print(f"DataFrame después: {result.shape[0]} filas, {result.shape[1]} columnas")
        print(f"Columnas: {list(result.columns)}")
        
        # Verificaciones
        assert len(result) == len(df), "El número de filas cambió"
        assert 'mcuentas_saldo' not in result.columns, "mcuentas_saldo no se eliminó"
        assert 'rf_001_001' not in result.columns, "rf_001_001 no se eliminó"
        assert 'mprestamos_total' in result.columns, "mprestamos_total se eliminó incorrectamente"
        assert 'rf_001_002' in result.columns, "rf_001_002 se eliminó incorrectamente"
        assert result.shape[1] == df.shape[1] - 2, "El número de columnas no es correcto"
        
        print("✅ Test 1 PASS: Features eliminadas correctamente")
        
    finally:
        # Limpiar archivo temporal
        os.unlink(csv_path)
    
    return True


def test_feature_dropper_without_csv():
    """
    Test del FeatureDropperTransformer sin CSV (no debería eliminar nada).
    """
    print("\n" + "="*60)
    print("Test de FeatureDropperTransformer sin CSV")
    print("="*60)
    
    # Crear datos sintéticos
    np.random.seed(42)
    df = pd.DataFrame({
        'mcuentas_saldo': np.random.rand(10),
        'mprestamos_total': np.random.rand(10),
    })
    
    print(f"\nDataFrame original: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Test con csv_path=None
    print("\n--- Test: csv_path=None ---")
    transformer = FeatureDropperTransformer(csv_path=None)
    transformer.fit(df)
    result = transformer.transform(df)
    
    assert df.equals(result), "El DataFrame cambió cuando no debería"
    print("✅ Test PASS: No se eliminó ninguna feature (csv_path=None)")
    
    return True


def test_feature_dropper_csv_not_found():
    """
    Test del FeatureDropperTransformer con CSV inexistente (no debería fallar).
    """
    print("\n" + "="*60)
    print("Test de FeatureDropperTransformer con CSV inexistente")
    print("="*60)
    
    # Crear datos sintéticos
    np.random.seed(42)
    df = pd.DataFrame({
        'mcuentas_saldo': np.random.rand(10),
        'mprestamos_total': np.random.rand(10),
    })
    
    print(f"\nDataFrame original: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Test con CSV inexistente
    print("\n--- Test: CSV inexistente ---")
    transformer = FeatureDropperTransformer(csv_path='/path/que/no/existe.csv')
    transformer.fit(df)
    result = transformer.transform(df)
    
    assert df.equals(result), "El DataFrame cambió cuando no debería"
    print("✅ Test PASS: No se eliminó ninguna feature (CSV inexistente)")
    
    return True


def test_feature_dropper_empty_csv():
    """
    Test del FeatureDropperTransformer con CSV vacío.
    """
    print("\n" + "="*60)
    print("Test de FeatureDropperTransformer con CSV vacío")
    print("="*60)
    
    # Crear datos sintéticos
    np.random.seed(42)
    df = pd.DataFrame({
        'mcuentas_saldo': np.random.rand(10),
        'mprestamos_total': np.random.rand(10),
    })
    
    print(f"\nDataFrame original: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Crear un CSV temporal vacío (solo header)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
        f.write("feature\n")
    
    print(f"\nCSV temporal vacío creado: {csv_path}")
    
    try:
        transformer = FeatureDropperTransformer(csv_path=csv_path)
        transformer.fit(df)
        result = transformer.transform(df)
        
        assert df.equals(result), "El DataFrame cambió cuando no debería"
        print("✅ Test PASS: No se eliminó ninguna feature (CSV vacío)")
        
    finally:
        os.unlink(csv_path)
    
    return True


def test_feature_dropper_all_features_missing():
    """
    Test cuando todas las features del CSV no existen en el DataFrame.
    """
    print("\n" + "="*60)
    print("Test con features que no existen en el DataFrame")
    print("="*60)
    
    # Crear datos sintéticos
    np.random.seed(42)
    df = pd.DataFrame({
        'mcuentas_saldo': np.random.rand(10),
        'mprestamos_total': np.random.rand(10),
    })
    
    print(f"\nDataFrame original: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    
    # Crear un CSV con features que no existen
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
        f.write("feature\n")
        f.write("feature_inexistente_1\n")
        f.write("feature_inexistente_2\n")
    
    print(f"\nCSV con features inexistentes: {csv_path}")
    
    try:
        transformer = FeatureDropperTransformer(csv_path=csv_path)
        transformer.fit(df)
        result = transformer.transform(df)
        
        assert df.equals(result), "El DataFrame cambió cuando no debería"
        print("✅ Test PASS: No se eliminó ninguna feature (todas inexistentes)")
        
    finally:
        os.unlink(csv_path)
    
    return True


if __name__ == "__main__":
    print("Ejecutando tests del FeatureDropperTransformer...\n")
    
    success_with_csv = test_feature_dropper_with_csv()
    success_without_csv = test_feature_dropper_without_csv()
    success_csv_not_found = test_feature_dropper_csv_not_found()
    success_empty_csv = test_feature_dropper_empty_csv()
    success_all_missing = test_feature_dropper_all_features_missing()
    
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"Test con CSV: {'✓ PASS' if success_with_csv else '✗ FAIL'}")
    print(f"Test sin CSV: {'✓ PASS' if success_without_csv else '✗ FAIL'}")
    print(f"Test CSV inexistente: {'✓ PASS' if success_csv_not_found else '✗ FAIL'}")
    print(f"Test CSV vacío: {'✓ PASS' if success_empty_csv else '✗ FAIL'}")
    print(f"Test features inexistentes: {'✓ PASS' if success_all_missing else '✗ FAIL'}")
    
    if all([success_with_csv, success_without_csv, success_csv_not_found, 
            success_empty_csv, success_all_missing]):
        print("\n🎉 TODOS LOS TESTS PASARON EXITOSAMENTE!")
        sys.exit(0)
    else:
        print("\n❌ ALGUNOS TESTS FALLARON.")
        sys.exit(1)

