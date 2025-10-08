"""
Módulo de manejo de datos: carga, preprocessing y división
"""
from .loading import cargar_datos, load_and_prepare_data, calcular_pesos_clase
from .preprocessing import sample_dataset_estratificado, obtener_columnas_por_tipo, transformar_columnas_categoricas
from .splitting import split_train_test_eval, split_train_test_eval_legacy

__all__ = [
    # Loading
    'cargar_datos',
    'load_and_prepare_data', 
    'calcular_pesos_clase',
    # Preprocessing
    'sample_dataset_estratificado',
    'obtener_columnas_por_tipo',
    'transformar_columnas_categoricas',
    # Splitting
    'split_train_test_eval',
    'split_train_test_eval_legacy'
]
