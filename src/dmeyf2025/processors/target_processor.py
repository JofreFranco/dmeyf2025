from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from dmeyf2025.utils.decorators import save_data_decorator

class CreateTargetProcessor(BaseEstimator, TransformerMixin):
    """
    Procesador de sklearn para crear la clase ternaria de clientes.
    
    Este procesador identifica si un cliente:
    - CONTINUA: Sigue activo
    - BAJA+1: Se da de baja en el mes siguiente
    - BAJA+2: Se da de baja en dos meses
    
    Requiere que el DataFrame tenga las columnas 'foto_mes' y 'numero_de_cliente'.
    """
    
    def __init__(self):
        """
        Inicializa el procesador de crear target.
        """
        pass
    
    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> 'CreateTargetProcessor':
        """
        Método fit requerido por sklearn.
        """
        return self
    
    @save_data_decorator("data/competencia_01_target.csv")
    def transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """
        Calcula la clase ternaria para cada cliente en cada mes.
        
        Args:
            X: DataFrame con columnas 'foto_mes' y 'numero_de_cliente'
            
        Returns:
            DataFrame con la columna 'clase_ternaria' agregada
        """
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
        return df

    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        return self.transform(X)
    
class BinaryTargetProcessor(BaseEstimator, TransformerMixin):
    """
    Procesador de sklearn para crear la clase binaria de clientes.
    """
    def __init__(self, positive_classes: list = ["BAJA+1"]):
        self.positive_classes = positive_classes
    
    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> 'BinaryTargetProcessor':
        return self
    
    def transform(self, X: pd.DataFrame, y) -> pd.DataFrame:

        clase_binaria = np.array([1 if item in self.positive_classes else 0 for item in y])
        return X, clase_binaria
    
    def get_positive_classes(self) -> list:
        return self.positive_classes
    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        return self.transform(X, y)