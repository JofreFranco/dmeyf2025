from typing import Any, Optional
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CreateTargetProcessor(BaseEstimator, TransformerMixin):
    """
    Procesador de sklearn para crear la clase ternaria de clientes.
    
    Este procesador identifica si un cliente:
    - CONTINUA: Sigue activo
    - BAJA+1: Se da de baja en el mes siguiente
    - BAJA+2: Se da de baja en dos meses
    
    Requiere que el DataFrame tenga las columnas 'foto_mes' y 'numero_de_cliente'.
    """
    
    def __init__(self, target_path: str):
        """
        Inicializa el procesador de crear target.
        
        Args:
            target_path: Path completo donde guardar/leer el archivo con el target calculado.
        """
        self.target_path = target_path
    
    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> 'CreateTargetProcessor':
        """
        Método fit requerido por sklearn.
        """
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """
        Calcula la clase ternaria para cada cliente en cada mes.
        
        Args:
            X: DataFrame con columnas 'foto_mes' y 'numero_de_cliente'
            
        Returns:
            DataFrame con la columna 'clase_ternaria' agregada
        """
        # Si el archivo con targets ya existe, leerlo
        if os.path.exists(self.target_path):
            return pd.read_csv(self.target_path)
        
        if "clase_ternaria" in X.columns:
            return X
        df = X.copy()
        df_clientes = df[["foto_mes", "numero_de_cliente"]].copy()
        meses = df_clientes['foto_mes'].unique()
        df_clientes["clase_ternaria"] = "CONTINUA"
        
        for n, mes in enumerate(meses):
            if n < len(meses) - 1:
                baja_1 = set(df_clientes[(df_clientes['foto_mes'] == mes)]["numero_de_cliente"]) - set(df_clientes[(df_clientes['foto_mes'] == meses[n+1])]["numero_de_cliente"])
                df_clientes.loc[(df_clientes['numero_de_cliente'].isin(baja_1)) & (df_clientes['foto_mes'] == mes), 'clase_ternaria'] = "BAJA+1"

            if n < len(meses) - 2:
                baja_2 = (set(df_clientes[(df_clientes['foto_mes'] == meses[n+1])]["numero_de_cliente"]) & set(df_clientes[(df_clientes['foto_mes'] == mes)]["numero_de_cliente"])) - set(df_clientes[(df_clientes['foto_mes'] == meses[n+2])]["numero_de_cliente"])
                df_clientes.loc[(df_clientes['numero_de_cliente'].isin(baja_2)) & (df_clientes['foto_mes'] == mes), 'clase_ternaria'] = "BAJA+2"
        
        df["clase_ternaria"] = df_clientes["clase_ternaria"]
        
        # Guardar el resultado en el archivo
        df.to_csv(self.target_path, index=False)
        
        return df

    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None):
        return self.transform(X)
    
class BinaryTargetProcessor(BaseEstimator, TransformerMixin):
    """
    Procesador de sklearn para crear la clase binaria de clientes.
    """
    def __init__(self, positive_classes: list = ["BAJA+1", "BAJA+2"], weight: dict = {"BAJA+1": 1, "BAJA+2": 1.00002, "CONTINUA": 1}):
        self.positive_classes = positive_classes
        self.weight = weight
    def fit(self, X: pd.DataFrame, y: Optional[Any] = None):
        return self
    
    def transform(self, X: pd.DataFrame, y):

        clase_binaria = np.array([1 if item in self.positive_classes else 0 for item in y])
        y_weight = np.array([self.weight[item] for item in y])

        return X, clase_binaria, y_weight
    
    def get_positive_classes(self) -> list:
        return self.positive_classes
    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        return self.transform(X, y)