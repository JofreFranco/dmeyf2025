"""
Utilidades para calcular y guardar la importancia de features de modelos
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def save_feature_importance_from_models(models, experiment_path, top_n=30):
    """
    Guarda la importancia de features promediando sobre un ensemble de modelos LightGBM.
    
    Genera:
    1. Un archivo CSV con todas las features y su importancia promedio ordenadas
    2. Un gráfico con las top N features más importantes
    
    Parameters:
    -----------
    models : list of lgb.Booster
        Lista de modelos LightGBM entrenados (ensemble)
    experiment_path : Path o str
        Ruta donde guardar los archivos
    top_n : int, default=30
        Número de features más importantes a mostrar en el gráfico
    """
    try:
        experiment_path = Path(experiment_path)
        
        logger.info("\n" + "="*70)
        logger.info("🔍 Calculando importancia de features del ensemble...")
        logger.info("="*70)
        logger.info(f"Número de modelos en el ensemble: {len(models)}")
        
        # Obtener feature names del primer modelo
        feature_names = models[0].feature_name()
        
        # Recopilar importancias de todos los modelos
        all_importances = []
        for i, model in enumerate(models):
            importance = model.feature_importance(importance_type='gain')
            all_importances.append(importance)
            logger.info(f"  Modelo {i+1}: {len(importance)} features")
        
        # Promediar importancias
        avg_importance = np.mean(all_importances, axis=0)
        std_importance = np.std(all_importances, axis=0)
        
        # Crear DataFrame con importancias
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': np.round(avg_importance, 3),
            'importance_std': np.round(std_importance, 3)
        })
        
        # Ordenar por importancia descendente
        importance_df = importance_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
        
        # Guardar CSV con todas las features
        csv_path = experiment_path / 'feature_importance.csv'
        importance_df.to_csv(csv_path, index=False)
        logger.info(f"💾 Importancia de features guardada en: {csv_path}")
        
        # Crear gráfico con top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['importance_mean'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importancia Promedio (Gain)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Features más importantes (Ensemble)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Guardar gráfico
        plot_path = experiment_path / 'feature_importance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 Gráfico de importancia guardado en: {plot_path}")
        
        # Log de las top 10 features
        logger.info("\n" + "="*70)
        logger.info("🏆 Top 10 Features más importantes (Ensemble):")
        logger.info("="*70)
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"{idx+1:2d}. {row['feature']:40s} | Importancia: {row['importance_mean']:,.0f} ± {row['importance_std']:,.0f}")
        logger.info("="*70)
        
        return importance_df
        
    except Exception as e:
        logger.error(f"❌ Error al guardar importancia de features: {e}")
        return None
