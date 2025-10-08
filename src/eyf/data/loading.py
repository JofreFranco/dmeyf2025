"""
Funciones para carga y preparación de datos
"""
import os
import pandas as pd

from ..utils.data_dict import PESO_BAJA_2, PESO_BAJA_1, PESO_CONTINUA
from .clase_ternaria import calcular_clase_ternaria, calcular_clase_binaria


def cargar_datos(raw_data_path, target_data_path=None, recalcular=False):
    """
    Carga los datos desde archivo CSV y calcula la clase ternaria si es necesario.
    
    Parameters:
    -----------
    raw_data_path : str
        Ruta al archivo de datos crudos
    target_data_path : str, optional
        Ruta donde guardar/cargar los datos con target calculado
    recalcular : bool, default=False
        Si True, fuerza el recálculo del target aunque ya exista
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con los datos y las clases calculadas
    """
    
    # Si no se proporciona ruta de target, usar la misma ruta pero con sufijo _target
    if target_data_path is None:
        base_path = raw_data_path.rsplit('.', 1)[0]
        target_data_path = f"{base_path}_target.csv"
    
    # Verificar si ya existe el archivo con target calculado
    if not recalcular and os.path.exists(target_data_path):
        print("Cargando datos con target existente...")
        df = pd.read_csv(target_data_path)
    else:
        print("Calculando target desde datos crudos...")
        # Cargar datos crudos
        print(f"Cargando datos desde: {raw_data_path}")
        df = pd.read_csv(raw_data_path)
        print(f"Datos cargados: {df.shape[0]} filas y {df.shape[1]} columnas")

        # Calcular la clase ternaria
        print("Calculando clase ternaria...")
        df = calcular_clase_ternaria(df)
        
        # Calcular la clase binaria
        print("Calculando clase binaria...")
        df = calcular_clase_binaria(df)

        # Verificar la distribución de clases
        print("\nDistribución de la clase ternaria:")
        print(df['clase_ternaria'].value_counts())
        print("\nPorcentajes:")
        print(df['clase_ternaria'].value_counts(normalize=True) * 100)
        
        print("\nDistribución de la clase binaria:")
        print(df['clase_binaria'].value_counts())
        print("\nPorcentajes:")
        print(df['clase_binaria'].value_counts(normalize=True) * 100)
        
        # Guardar los datos con target calculado
        print(f"Guardando datos con target en: {target_data_path}")
        df.to_csv(target_data_path, index=False)
    
    return df


def calcular_pesos_clase(df):
    """
    Calcula los pesos de clase basado en la clase ternaria.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con la columna 'clase_ternaria'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con la nueva columna 'clase_peso'
    
    Notes:
    ------
    Los pesos son:
    - BAJA+2: 1.00002 (clase más importante)
    - BAJA+1: 1.00001 (clase intermedia)  
    - CONTINUA: 1.0 (clase base)
    """
    df = df.copy()
    
    # Mapear clases a pesos
    peso_map = {
        'BAJA+2': PESO_BAJA_2,
        'BAJA+1': PESO_BAJA_1,
        'CONTINUA': PESO_CONTINUA
    }
    
    df['clase_peso'] = df['clase_ternaria'].map(peso_map)
    
    # Verificar que no hay valores nulos
    if df['clase_peso'].isnull().any():
        print("⚠️ Advertencia: Se encontraron valores nulos en clase_peso")
        print("Clases encontradas:", df['clase_ternaria'].unique())
    
    return df


def load_and_prepare_data(raw_data_path, target_data_path=None, preprocessor=None):
    """
    Carga datos, calcula targets, aplica pesos y preprocessing.
    
    Parameters:
    -----------
    raw_data_path : str
        Ruta a datos crudos
    target_data_path : str, optional
        Ruta a datos con target calculado
    preprocessor : sklearn transformer, optional
        Preprocessor a aplicar
        
    Returns:
    --------
    pd.DataFrame
        DataFrame procesado y listo para usar
    """
    # Cargar datos
    df = cargar_datos(raw_data_path, target_data_path)
    
    # Calcular pesos
    df = calcular_pesos_clase(df)
    
    # Aplicar preprocessor
    if preprocessor is not None:
        preprocessor.fit(df)
        df = preprocessor.transform(df)
    
    return df
