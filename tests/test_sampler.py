import pandas as pd
import numpy as np
import sys
import os

# Agregar el directorio src al path para importar los módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dmeyf2025.processors.sampler import SamplerProcessor


def load_dataset():
    """Carga el dataset de competencia."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'competencia_01_target.csv')
    print(f"Cargando dataset desde: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
    return df


def print_dataset_info(df, title="Dataset"):
    """Imprime información sobre el dataset."""
    print(f"\n{title}:")
    print(f"  - Total registros: {df.shape[0]}")
    print(f"  - Clientes únicos: {df['numero_de_cliente'].nunique() if 'numero_de_cliente' in df.columns else 'N/A'}")
    print(f"  - Meses únicos: {df['foto_mes'].nunique() if 'foto_mes' in df.columns else 'N/A'}")
    if 'foto_mes' in df.columns:
        print(f"  - Rango meses: {df['foto_mes'].min()} - {df['foto_mes'].max()}")
    print(f"  - Clase BAJA+1: {(df['clase_ternaria'] == 'BAJA+1').sum()}")
    print(f"  - Clase BAJA+2: {(df['clase_ternaria'] == 'BAJA+2').sum()}")
    print(f"  - Clase CONTINUA: {(df['clase_ternaria'] == 'CONTINUA').sum()}")


def test_random_sampling():
    """Test del método random_transform."""
    print("\n" + "="*80)
    print("TEST 1: Random Sampling")
    print("="*80)
    
    df = load_dataset()
    print_dataset_info(df, "Dataset original")
    
    # Preparar X e y
    X = df.drop(columns=['clase_ternaria'])
    y = (df['clase_ternaria'] == 'BAJA+2').astype(int)
    
    # Test con diferentes ratios
    ratios = [1.0, 0.5, 0.1, 0.01]
    
    all_pass = True
    for ratio in ratios:
        print(f"\n--- Test con sample_ratio={ratio} ---")
        sampler = SamplerProcessor(
            sample_ratio=ratio, 
            random_state=42, 
            sampling_type="random"
        )
        
        X_sampled, y_sampled = sampler.transform(X.copy(), y.copy())
        
        # Contar clases
        n_positives = y_sampled.sum()
        n_negatives = (y_sampled == 0).sum()
        
        print(f"  Registros originales: {len(X)}")
        print(f"  Registros muestreados: {len(X_sampled)}")
        print(f"  Ratio real: {len(X_sampled) / len(X):.4f}")
        print(f"  Clase positiva: {n_positives}")
        print(f"  Clase negativa: {n_negatives}")
        
        # Verificación: Si ratio < 1, debe reducirse el dataset
        if ratio < 1.0:
            if len(X_sampled) < len(X):
                print("  ✅ PASS: Dataset reducido correctamente")
            else:
                print("  ❌ FAIL: Dataset no se redujo")
                all_pass = False
        else:
            if len(X_sampled) == len(X):
                print("  ✅ PASS: Dataset completo (ratio=1.0)")
            else:
                print("  ❌ FAIL: Dataset modificado con ratio=1.0")
                all_pass = False
    
    return all_pass


def test_volta_sampling():
    """Test del método volta_transform."""
    print("\n" + "="*80)
    print("TEST 2: Volta Sampling (cliente-based)")
    print("="*80)
    
    df = load_dataset()
    print_dataset_info(df, "Dataset original")
    
    X = df.drop(columns=['clase_ternaria'])
    y = (df['clase_ternaria'] == 'BAJA+2').astype(int)
    
    original_clients = X['numero_de_cliente'].nunique()
    
    # Test con diferentes ratios
    ratios = [0.5, 0.1]
    
    all_pass = True
    for ratio in ratios:
        print(f"\n--- Test con sample_ratio={ratio} ---")
        sampler = SamplerProcessor(
            sample_ratio=ratio, 
            random_state=42, 
            sampling_type="volta"
        )
        
        X_sampled, y_sampled = sampler.transform(X.copy(), y.copy())
        
        sampled_clients = X_sampled['numero_de_cliente'].nunique()
        expected_clients = int(original_clients * ratio)
        
        print(f"  Clientes originales: {original_clients}")
        print(f"  Clientes muestreados: {sampled_clients}")
        print(f"  Clientes esperados: {expected_clients}")
        print(f"  Registros muestreados: {len(X_sampled)}")
        print(f"  Ratio de registros: {len(X_sampled) / len(X):.4f}")
        
        # Verificación: El número de clientes debe ser aproximadamente ratio * original
        tolerance = 0.05 * original_clients
        if abs(sampled_clients - expected_clients) < tolerance:
            print("  ✅ PASS: Número de clientes correcto")
        else:
            print("  ❌ FAIL: Número de clientes incorrecto")
            all_pass = False
        
        # Verificar que se mantiene toda la historia de los clientes seleccionados
        sampled_client_ids = X_sampled['numero_de_cliente'].unique()
        for client_id in np.random.choice(sampled_client_ids, min(5, len(sampled_client_ids)), replace=False):
            original_records = len(X[X['numero_de_cliente'] == client_id])
            sampled_records = len(X_sampled[X_sampled['numero_de_cliente'] == client_id])
            if original_records != sampled_records:
                print(f"  ❌ FAIL: Cliente {client_id} no tiene toda su historia")
                all_pass = False
                break
        else:
            print("  ✅ PASS: Historia completa de clientes preservada")
    
    return all_pass


def test_linear_sampling():
    """Test del método linear_transform."""
    print("\n" + "="*80)
    print("TEST 3: Linear Sampling")
    print("="*80)
    
    df = load_dataset()
    print_dataset_info(df, "Dataset original")
    
    X = df.drop(columns=['clase_ternaria'])
    y = (df['clase_ternaria'] == 'BAJA+2').astype(int)
    
    # Test con diferentes velocidades
    speeds = [0.05, 0.1, 0.2]
    
    all_pass = True
    for speed in speeds:
        print(f"\n--- Test con speed={speed} ---")
        sampler = SamplerProcessor(
            random_state=42, 
            sampling_type="linear",
            speed=speed
        )
        
        X_sampled, y_sampled = sampler.transform(X.copy(), y.copy())
        
        print(f"  Registros originales: {len(X)}")
        print(f"  Registros muestreados: {len(X_sampled)}")
        print(f"  Ratio real: {len(X_sampled) / len(X):.4f}")
        
        # Analizar distribución por mes
        print("\n  Distribución por mes:")
        unique_months = sorted(X_sampled['foto_mes'].unique(), reverse=True)
        
        for idx, month in enumerate(unique_months[:5]):  # Mostrar solo los primeros 5
            month_original = len(X[X['foto_mes'] == month])
            month_sampled = len(X_sampled[X_sampled['foto_mes'] == month])
            expected_ratio = max(0.0, 1.0 - speed * idx)
            actual_ratio = month_sampled / month_original if month_original > 0 else 0
            
            print(f"    Mes {month} (x={idx}): {month_sampled}/{month_original} registros, "
                  f"ratio={actual_ratio:.4f}, esperado≈{expected_ratio:.4f}")
        
        # Verificación: el mes más reciente debe tener ratio cercano a 1.0
        newest_month = unique_months[0]
        newest_original = len(X[X['foto_mes'] == newest_month])
        newest_sampled = len(X_sampled[X_sampled['foto_mes'] == newest_month])
        newest_ratio = newest_sampled / newest_original
        
        if abs(newest_ratio - 1.0) < 0.05:
            print(f"\n  ✅ PASS: Mes más reciente tiene ratio≈1.0 ({newest_ratio:.4f})")
        else:
            print(f"\n  ❌ FAIL: Mes más reciente tiene ratio incorrecto ({newest_ratio:.4f})")
            all_pass = False
    
    return all_pass


def test_exponential_sampling():
    """Test del método exponential_transform."""
    print("\n" + "="*80)
    print("TEST 4: Exponential Sampling")
    print("="*80)
    
    df = load_dataset()
    print_dataset_info(df, "Dataset original")
    
    X = df.drop(columns=['clase_ternaria'])
    y = (df['clase_ternaria'] == 'BAJA+2').astype(int)
    
    # Test con diferentes velocidades
    speeds = [0.05, 0.1, 0.3]
    
    all_pass = True
    for speed in speeds:
        print(f"\n--- Test con speed={speed} ---")
        sampler = SamplerProcessor(
            random_state=42, 
            sampling_type="exponential",
            speed=speed
        )
        
        X_sampled, y_sampled = sampler.transform(X.copy(), y.copy())
        
        print(f"  Registros originales: {len(X)}")
        print(f"  Registros muestreados: {len(X_sampled)}")
        print(f"  Ratio real: {len(X_sampled) / len(X):.4f}")
        
        # Analizar distribución por mes
        print("\n  Distribución por mes:")
        unique_months = sorted(X_sampled['foto_mes'].unique(), reverse=True)
        
        for idx, month in enumerate(unique_months[:5]):  # Mostrar solo los primeros 5
            month_original = len(X[X['foto_mes'] == month])
            month_sampled = len(X_sampled[X_sampled['foto_mes'] == month])
            expected_ratio = np.exp(-speed * idx)
            actual_ratio = month_sampled / month_original if month_original > 0 else 0
            
            print(f"    Mes {month} (x={idx}): {month_sampled}/{month_original} registros, "
                  f"ratio={actual_ratio:.4f}, esperado≈{expected_ratio:.4f}")
        
        # Verificación: el mes más reciente debe tener ratio cercano a 1.0
        newest_month = unique_months[0]
        newest_original = len(X[X['foto_mes'] == newest_month])
        newest_sampled = len(X_sampled[X_sampled['foto_mes'] == newest_month])
        newest_ratio = newest_sampled / newest_original
        
        if abs(newest_ratio - 1.0) < 0.05:
            print(f"\n  ✅ PASS: Mes más reciente tiene ratio≈1.0 ({newest_ratio:.4f})")
        else:
            print(f"\n  ❌ FAIL: Mes más reciente tiene ratio incorrecto ({newest_ratio:.4f})")
            all_pass = False
        
        # Verificar decrecimiento exponencial
        ratios = []
        for idx, month in enumerate(unique_months[:5]):
            month_original = len(X[X['foto_mes'] == month])
            month_sampled = len(X_sampled[X_sampled['foto_mes'] == month])
            ratio = month_sampled / month_original if month_original > 0 else 0
            ratios.append(ratio)
        
        # Los ratios deben decrecer más rápido al principio (exponencial)
        if len(ratios) >= 3:
            decay_1 = ratios[0] - ratios[1]
            decay_2 = ratios[1] - ratios[2]
            if decay_1 >= decay_2:
                print("  ✅ PASS: Decrecimiento exponencial verificado")
            else:
                print("  ⚠️  WARNING: Decrecimiento no parece exponencial")
    
    return all_pass


def test_comparison():
    """Comparación visual de los 4 métodos."""
    print("\n" + "="*80)
    print("TEST 5: Comparación de métodos")
    print("="*80)
    
    df = load_dataset()
    X = df.drop(columns=['clase_ternaria'])
    y = (df['clase_ternaria'] == 'BAJA+2').astype(int)
    
    methods = [
        ("random", {"sample_ratio": 0.5}),
        ("volta", {"sample_ratio": 0.5}),
        ("linear", {"speed": 0.1}),
        ("exponential", {"speed": 0.1}),
    ]
    
    print(f"\nDataset original: {len(X)} registros\n")
    
    results = []
    for method_name, params in methods:
        sampler = SamplerProcessor(
            random_state=42, 
            sampling_type=method_name,
            **params
        )
        
        X_sampled, y_sampled = sampler.transform(X.copy(), y.copy())
        
        ratio = len(X_sampled) / len(X)
        results.append({
            "Método": method_name,
            "Parámetros": str(params),
            "Registros": len(X_sampled),
            "Ratio": f"{ratio:.4f}",
            "Clase +": y_sampled.sum(),
            "Clase 0": (y_sampled == 0).sum(),
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    return True


def test_speed_calculation():
    """Test del cálculo de speed para lograr un ratio objetivo."""
    print("\n" + "="*80)
    print("TEST 6: Cálculo de speed para ratio objetivo")
    print("="*80)
    
    df = load_dataset()
    X = df.drop(columns=['clase_ternaria'])
    y = (df['clase_ternaria'] == 'BAJA+2').astype(int)
    
    print(f"\nDataset original: {len(X)} registros")
    print(f"Meses disponibles: {sorted(X['foto_mes'].unique())}")
    
    # Test para diferentes ratios objetivo
    target_ratios = [0.9, 0.7, 0.5, 0.3]
    
    all_pass = True
    
    for target_ratio in target_ratios:
        print(f"\n{'='*60}")
        print(f"Objetivo: Ratio equivalente = {target_ratio:.2f} ({target_ratio*100:.0f}%)")
        print(f"{'='*60}")
        
        results = []
        
        for sampling_type in ["linear", "exponential"]:
            # Calcular speed necesario
            speed = SamplerProcessor.calculate_speed_for_target_ratio(
                X, target_ratio, sampling_type, tolerance=0.01
            )
            
            # Verificar el speed calculado
            calculated_ratio = SamplerProcessor.calculate_equivalent_ratio(
                X, speed, sampling_type
            )
            
            # Aplicar el sampling para verificar
            sampler = SamplerProcessor(
                random_state=42,
                sampling_type=sampling_type,
                speed=speed
            )
            X_sampled, y_sampled = sampler.transform(X.copy(), y.copy())
            actual_ratio = len(X_sampled) / len(X)
            
            results.append({
                "Tipo": sampling_type,
                "Speed": f"{speed:.6f}",
                "Ratio calculado": f"{calculated_ratio:.4f}",
                "Ratio real": f"{actual_ratio:.4f}",
                "Error": f"{abs(actual_ratio - target_ratio):.4f}",
                "Registros": len(X_sampled),
            })
            
            # Verificar que el error es pequeño
            error = abs(actual_ratio - target_ratio)
            if error < 0.02:  # Tolerancia de 2%
                print(f"  ✅ {sampling_type}: speed={speed:.6f} -> ratio={actual_ratio:.4f}")
            else:
                print(f"  ❌ {sampling_type}: speed={speed:.6f} -> ratio={actual_ratio:.4f} (error={error:.4f})")
                all_pass = False
        
        print("\n  Detalles:")
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
    
    return all_pass


def test_calculate_ratio_for_speed():
    """Test del cálculo de ratio equivalente dado un speed."""
    print("\n" + "="*80)
    print("TEST 7: Cálculo de ratio equivalente para diferentes speeds")
    print("="*80)
    
    df = load_dataset()
    X = df.drop(columns=['clase_ternaria'])
    
    print(f"\nDataset: {len(X)} registros")
    print(f"Meses: {sorted(X['foto_mes'].unique())}")
    
    speeds = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    print("\n" + "-"*80)
    print("LINEAR SAMPLING")
    print("-"*80)
    print(f"{'Speed':<10} {'Ratio equiv':<15} {'% Dataset':<15}")
    print("-"*80)
    
    for speed in speeds:
        ratio = SamplerProcessor.calculate_equivalent_ratio(X, speed, "linear")
        print(f"{speed:<10.2f} {ratio:<15.4f} {ratio*100:<15.1f}%")
    
    print("\n" + "-"*80)
    print("EXPONENTIAL SAMPLING")
    print("-"*80)
    print(f"{'Speed':<10} {'Ratio equiv':<15} {'% Dataset':<15}")
    print("-"*80)
    
    for speed in speeds:
        ratio = SamplerProcessor.calculate_equivalent_ratio(X, speed, "exponential")
        print(f"{speed:<10.2f} {ratio:<15.4f} {ratio*100:<15.1f}%")
    
    return True


if __name__ == "__main__":
    print("Ejecutando tests del SamplerProcessor...\n")
    
    try:
        # Ejecutar tests
        test1_pass = test_random_sampling()
        test2_pass = test_volta_sampling()
        test3_pass = test_linear_sampling()
        test4_pass = test_exponential_sampling()
        test5_pass = test_comparison()
        test6_pass = test_speed_calculation()
        test7_pass = test_calculate_ratio_for_speed()
        
        # Resumen final
        print("\n" + "="*80)
        print("RESUMEN FINAL")
        print("="*80)
        print(f"Test 1 - Random Sampling:          {'✓ PASS' if test1_pass else '✗ FAIL'}")
        print(f"Test 2 - Volta Sampling:           {'✓ PASS' if test2_pass else '✗ FAIL'}")
        print(f"Test 3 - Linear Sampling:          {'✓ PASS' if test3_pass else '✗ FAIL'}")
        print(f"Test 4 - Exponential Sampling:     {'✓ PASS' if test4_pass else '✗ FAIL'}")
        print(f"Test 5 - Comparación:              {'✓ PASS' if test5_pass else '✗ FAIL'}")
        print(f"Test 6 - Cálculo de Speed:         {'✓ PASS' if test6_pass else '✗ FAIL'}")
        print(f"Test 7 - Ratio por Speed:          {'✓ PASS' if test7_pass else '✗ FAIL'}")
        
        all_pass = test1_pass and test2_pass and test3_pass and test4_pass and test5_pass and test6_pass and test7_pass
        
        if all_pass:
            print("\n🎉 TODOS LOS TESTS PASARON EXITOSAMENTE!")
            sys.exit(0)
        else:
            print("\n❌ ALGUNOS TESTS FALLARON.")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ ERROR durante la ejecución de tests: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

