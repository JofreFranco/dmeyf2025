import pandas as pd
import numpy as np
import sys
import os

# Agregar el directorio src al path para importar los módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dmeyf2025.processors.feature_processors import PeriodStatsTransformer


def test_period_stats_transformer():
    """
    Función de test para PeriodStatsTransformer.
    Toma el dataset completo, selecciona 100 clientes distintos y aplica el transformador.
    """
    
    print("Iniciando test del PeriodStatsTransformer...")
    
    # Cargar el dataset usando path relativo desde el directorio del proyecto
    print("Cargando dataset...")
    try:
        # Buscar el archivo desde el directorio del proyecto
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, 'data', 'competencia_01_crudo.csv')
        df = pd.read_csv(data_path)
        print(f"Dataset cargado desde: {data_path}")
        print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return False
    
    # Seleccionar 100 clientes distintos
    unique_clients = df['numero_de_cliente'].unique()
    print(f"Total de clientes únicos: {len(unique_clients)}")
    
    # Seleccionar 100 clientes aleatorios
    np.random.seed(42)  # Para reproducibilidad
    selected_clients = np.random.choice(unique_clients, size=min(100, len(unique_clients)), replace=False)
    
    # Filtrar el dataset para estos clientes
    test_df = df[df['numero_de_cliente'].isin(selected_clients)].copy()
    print(f"Dataset de test: {test_df.shape[0]} filas para {len(selected_clients)} clientes")
    
    # Verificar que tenemos datos suficientes
    if test_df.empty:
        print("Error: No hay datos para los clientes seleccionados")
        return False
    
    # Mostrar información básica del dataset de test
    print(f"\nInformación del dataset de test:")
    print(f"- Columnas numéricas: {len(test_df.select_dtypes(include='number').columns)}")
    print(f"- Rango de foto_mes: {test_df['foto_mes'].min()} - {test_df['foto_mes'].max()}")
    print(f"- Registros por cliente (promedio): {test_df.groupby('numero_de_cliente').size().mean():.1f}")
    
    # Crear el transformador
    print(f"\nCreando PeriodStatsTransformer con periodo=12...")
    transformer = PeriodStatsTransformer(period=12)
    
    # Aplicar fit
    print("Aplicando fit...")
    transformer.fit(test_df)
    
    # Aplicar transform
    print("Aplicando transform...")
    try:
        transformed_df = transformer.transform(test_df)
        print(f"Transformación exitosa!")
        print(f"Dataset transformado: {transformed_df.shape[0]} filas, {transformed_df.shape[1]} columnas")
        
        # Calcular cuántas columnas nuevas se crearon
        original_cols = set(test_df.columns)
        new_cols = set(transformed_df.columns) - original_cols
        print(f"Columnas nuevas creadas: {len(new_cols)}")
        
        # Mostrar algunas estadísticas de las nuevas columnas
        print(f"\nEstadísticas de las nuevas columnas:")
        numeric_cols = [col for col in test_df.select_dtypes(include='number').columns 
                       if col not in ["foto_mes", "numero_de_cliente", "target", "label"]]
        
        for col in numeric_cols[:3]:  # Mostrar solo las primeras 3 columnas
            for stat in ['min', 'max', 'mean', 'median']:
                new_col = f'{col}_period12_{stat}'
                if new_col in transformed_df.columns:
                    non_null_count = transformed_df[new_col].notna().sum()
                    print(f"  {new_col}: {non_null_count} valores no nulos")
        
        # Verificar que no hay errores obvios
        print(f"\nVerificaciones:")
        print(f"- ¿Mantiene el mismo número de filas? {len(test_df) == len(transformed_df)}")
        print(f"- ¿Tiene más columnas? {len(transformed_df.columns) > len(test_df.columns)}")
        
        # Verificar que las columnas originales se mantienen
        original_preserved = all(col in transformed_df.columns for col in test_df.columns)
        print(f"- ¿Se mantienen las columnas originales? {original_preserved}")
        
        # Mostrar ejemplo de datos transformados
        print(f"\nEjemplo de datos transformados (primeras 5 filas, algunas columnas):")
        sample_cols = ['numero_de_cliente', 'foto_mes'] + list(new_cols)[:5]
        print(transformed_df[sample_cols].head())
        
        # Test adicional: Probar con filtro de meses específicos
        print(f"\n--- Test adicional: Con filtro de meses específicos ---")
        available_months = sorted(test_df['foto_mes'].unique())
        test_months = available_months[-2:]  # Últimos 2 meses disponibles
        
        print(f"Meses disponibles: {available_months}")
        print(f"Probando con meses: {test_months}")
        
        transformer_filtered = PeriodStatsTransformer(period=12, months=test_months)
        filtered_df = transformer_filtered.transform(test_df)
        
        print(f"Dataset filtrado: {filtered_df.shape[0]} filas")
        print(f"Meses en resultado filtrado: {sorted(filtered_df['foto_mes'].unique())}")
        
        # Verificar que solo contiene los meses especificados
        actual_months = set(filtered_df['foto_mes'].unique())
        expected_months = set(test_months)
        assert actual_months == expected_months, f"Esperado {expected_months}, obtenido {actual_months}"
        print("✅ Filtro de meses funciona correctamente con datos reales")
        
        return True
        
    except Exception as e:
        print(f"Error durante la transformación: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_period_stats_transformer_small():
    """
    Función de test más pequeña con datos sintéticos para verificación rápida.
    """
    print("\n" + "="*50)
    print("Test con datos sintéticos...")
    
    # Crear datos sintéticos
    np.random.seed(42)
    n_clients = 5
    n_months = 6
    
    data = []
    for client_id in range(1, n_clients + 1):
        for month in range(202101, 202101 + n_months):
            data.append({
                'numero_de_cliente': client_id,
                'foto_mes': month,
                'var1': np.random.normal(100, 20),
                'var2': np.random.normal(50, 10),
                'var3': np.random.normal(200, 30)
            })
    
    test_df = pd.DataFrame(data)
    print(f"Datos sintéticos creados: {test_df.shape}")
    
    # Test 1: Sin filtro de meses (comportamiento original)
    print("\n--- Test 1: Sin filtro de meses ---")
    transformer = PeriodStatsTransformer(period=3)
    transformed_df = transformer.transform(test_df)
    
    print(f"Datos transformados: {transformed_df.shape}")
    print(f"Columnas nuevas: {len(transformed_df.columns) - len(test_df.columns)}")
    print(f"Meses en resultado: {sorted(transformed_df['foto_mes'].unique())}")
    
    # Test 2: Con filtro de meses específicos
    print("\n--- Test 2: Con filtro de meses [202103, 202105] ---")
    transformer_filtered = PeriodStatsTransformer(period=3, months=[202103, 202105])
    transformed_filtered_df = transformer_filtered.transform(test_df)
    
    print(f"Datos transformados (filtrados): {transformed_filtered_df.shape}")
    print(f"Meses en resultado filtrado: {sorted(transformed_filtered_df['foto_mes'].unique())}")
    
    # Verificar que solo contiene los meses especificados
    expected_months = set([202103, 202105])
    actual_months = set(transformed_filtered_df['foto_mes'].unique())
    assert actual_months == expected_months, f"Esperado {expected_months}, obtenido {actual_months}"
    print("✅ Filtro de meses funciona correctamente")
    
    # Mostrar resultado
    print("\nResultado completo:")
    print(transformed_df[['numero_de_cliente', 'foto_mes', 'var1', 'var1_period3_mean', 'var1_period3_min']].head(10))
    
    print("\nResultado filtrado:")
    print(transformed_filtered_df[['numero_de_cliente', 'foto_mes', 'var1', 'var1_period3_mean', 'var1_period3_min']].head(10))
    
    return True


if __name__ == "__main__":
    print("Ejecutando tests del PeriodStatsTransformer...")
    
    # Test con datos sintéticos (rápido)
    success_small = test_period_stats_transformer_small()
    
    # Test con datos reales (más lento)
    success_real = test_period_stats_transformer()
    
    print("\n" + "="*50)
    print("RESUMEN DE TESTS:")
    print(f"Test con datos sintéticos: {'✓ PASS' if success_small else '✗ FAIL'}")
    print(f"Test con datos reales: {'✓ PASS' if success_real else '✗ FAIL'}")
    
    if success_small and success_real:
        print("\n🎉 Todos los tests pasaron exitosamente!")
    else:
        print("\n❌ Algunos tests fallaron.")
