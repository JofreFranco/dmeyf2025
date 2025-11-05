"""
Script de prueba para verificar la funcionalidad de importancia de features con modelos reales
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
from pathlib import Path
from dmeyf2025.utils.feature_importance import save_feature_importance_from_models

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

def test_feature_importance_with_ensemble():
    """
    Prueba la funcionalidad de importancia de features con un ensemble de modelos
    """
    print("="*70)
    print("PRUEBA DE IMPORTANCIA DE FEATURES CON ENSEMBLE")
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
    
    # Parámetros de prueba
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Crear directorio temporal para prueba
    test_dir = Path("test_feature_importance_v2_output")
    test_dir.mkdir(exist_ok=True)
    
    print(f"\nEntrenando ensemble de 3 modelos...")
    print(f"Datos: {n_samples} muestras, {n_features} features")
    
    # Entrenar ensemble de modelos
    models = []
    seeds = [42, 123, 456]
    
    for seed in seeds:
        params_copy = params.copy()
        params_copy['seed'] = seed
        
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            params_copy,
            train_data,
            num_boost_round=50,
            callbacks=[lgb.log_evaluation(0)]
        )
        models.append(model)
        print(f"  Modelo entrenado con seed {seed}")
    
    print(f"\nGenerando importancia de features del ensemble...")
    
    # Guardar importancia
    importance_df = save_feature_importance_from_models(
        models,
        test_dir,
        top_n=30
    )
    
    if importance_df is not None:
        print("\n✅ Prueba exitosa!")
        print(f"\nArchivos generados en: {test_dir}/")
        print(f"  - feature_importance.csv (con media y desviación estándar)")
        print(f"  - feature_importance.png")
        
        # Mostrar las top 5 features
        print("\nTop 5 Features:")
        for idx, row in importance_df.head(5).iterrows():
            print(f"  {idx+1}. {row['feature']:20s} | {row['importance_mean']:,.1f} ± {row['importance_std']:,.1f}")
    else:
        print("\n❌ La prueba falló")
    
    return importance_df

if __name__ == "__main__":
    test_feature_importance_with_ensemble()

