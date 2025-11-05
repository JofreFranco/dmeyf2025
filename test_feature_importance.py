"""
Script de prueba para verificar la funcionalidad de importancia de features
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from dmeyf2025.utils.feature_importance import calculate_and_save_feature_importance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

def test_feature_importance():
    """
    Prueba la funcionalidad de importancia de features con datos simulados
    """
    print("="*70)
    print("PRUEBA DE FUNCIONALIDAD DE IMPORTANCIA DE FEATURES")
    print("="*70)
    
    # Crear datos de prueba
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Generar features aleatorias
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    
    # Generar target binario
    y_train = np.random.randint(0, 2, n_samples)
    
    # Generar pesos
    w_train = np.ones(n_samples)
    
    # Parámetros de prueba (simulando los mejores parámetros)
    best_params = {
        'learning_rate': 0.01,
        'num_boost_round': 100,  # Pocas iteraciones para la prueba
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 42
    }
    
    # Crear directorio temporal para prueba
    test_dir = Path("test_feature_importance_output")
    test_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerando importancia de features en: {test_dir}")
    print(f"Datos de prueba: {n_samples} muestras, {n_features} features")
    
    # Calcular y guardar importancia
    importance_df = calculate_and_save_feature_importance(
        X_train, y_train, w_train,
        best_params,
        test_dir,
        top_n=30
    )
    
    if importance_df is not None:
        print("\n✅ Prueba exitosa!")
        print(f"\nArchivos generados en: {test_dir}/")
        print(f"  - feature_importance.csv")
        print(f"  - feature_importance.png")
        print("\nPuedes revisar los archivos generados.")
    else:
        print("\n❌ La prueba falló")
    
    return importance_df

if __name__ == "__main__":
    test_feature_importance()

