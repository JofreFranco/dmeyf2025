"""
Utilidades para visualización y gráficos
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sin display para entornos sin GUI
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance(model, feature_names=None, top_features=30, 
                          experiment_name="experiment", working_dir=".", debug_mode=False):
    """
    Grafica la importancia de features de un modelo LightGBM.
    
    Parameters:
    -----------
    model : lgb.Booster
        Modelo LightGBM entrenado
    feature_names : list, optional
        Nombres de las features. Si None, genera nombres genéricos
    top_features : int, default=30
        Número de features más importantes a mostrar
    experiment_name : str, default="experiment"
        Nombre del experimento para el archivo
    working_dir : str, default="."
        Directorio donde guardar el gráfico
    debug_mode : bool, default=False
        Si True, agrega sufijo _DEBUG al archivo
        
    Returns:
    --------
    str
        Ruta del archivo de imagen generado
    """
    # Obtener importancia de features (tipo gain)
    importance = model.feature_importance(importance_type='gain')
    
    # Generar nombres de features si no se proporcionan
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importance))]
    
    # Crear DataFrame con importancia
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Ordenar por importancia y tomar los top_features
    df_importance = df_importance.sort_values('importance', ascending=False).head(top_features)
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Crear figura con tamaño ajustado para 30 features
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Crear gráfico de barras horizontales
    bars = ax.barh(range(len(df_importance)), df_importance['importance'])
    
    # Personalizar gráfico
    ax.set_yticks(range(len(df_importance)))
    ax.set_yticklabels(df_importance['feature'], fontsize=8)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title(f'Top {top_features} Feature Importance - {experiment_name}', fontsize=14, pad=20)
    
    # Invertir orden del eje Y para mostrar la más importante arriba
    ax.invert_yaxis()
    
    # Agregar valores en las barras
    for i, (bar, value) in enumerate(zip(bars, df_importance['importance'])):
        ax.text(value + max(df_importance['importance']) * 0.01, 
               bar.get_y() + bar.get_height()/2, 
               f'{value:.0f}', 
               va='center', fontsize=7)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Generar nombre del archivo
    if debug_mode:
        filename = f"{experiment_name}_feature_importance_DEBUG.png"
    else:
        filename = f"{experiment_name}_feature_importance.png"
    
    filepath = os.path.join(working_dir, filename)
    
    # Guardar gráfico
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Cerrar figura para liberar memoria
    
    print(f"📊 Gráfico de importancia guardado: {filepath}")
    
    return filepath
