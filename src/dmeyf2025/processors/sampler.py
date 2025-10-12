import os
import logging
from typing import Any, Optional

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample




logger = logging.getLogger(__name__)


class SamplerProcessor(BaseEstimator, TransformerMixin):
    """
    Procesador de sklearn para realizar sampling aleatorio de un DataFrame.
    
    Este procesador recibe un DataFrame con una columna 'clase_ternaria' y realiza sampling aleatorio
    """
    
    def __init__(self, sample_ratio: float = 1, random_state: int = 42):
        """
        Inicializa el procesador de sampling.
        
        Args:
            sample_ratio: Proporción de muestreo para todas las clases (por defecto 1.0 = sin sampling)
            random_state: Semilla para reproducibilidad
        """
        self.sample_ratio = sample_ratio
        self.debug = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
        self.random_state = random_state
        
    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> 'SamplerProcessor':
        """
        Método fit requerido por sklearn.
        """
        return self

    def transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """
        Aplica sampling a la clase CONTINUA

        Args:
            X: features
            y: labels

        Returns:
            X_sampled, y_sampled: X e y muestreados
        """

        if self.sample_ratio >= 1.0:
            return X, y

        # Construimos un DataFrame que incluye y para facilitar muestreo conjunto
        df_sampled = X.copy()
        df_sampled['label'] = y

        # Separar clases
        continua_mask = df_sampled['label'] == 0
        other_classes = df_sampled[~continua_mask].copy()
        continua_cases = df_sampled[continua_mask].copy()
        n_continua_keep = int(len(continua_cases) * self.sample_ratio)

        # Hacer sampling de casos CONTINUA
        if n_continua_keep < len(continua_cases):
            continua_sampled = resample(
                continua_cases,
                n_samples=n_continua_keep,
                random_state=self.random_state,
                replace=False
            )
        else:
            continua_sampled = continua_cases

        # Combinar datasets
        df_final = pd.concat([other_classes, continua_sampled], ignore_index=True)

        logger.info(f"✅ Dataset final: {len(df_final)} registros")
        logger.info(f"   - Clase positiva: {(df_final['label'] == 1).sum()}")
        logger.info(f"   - Clase negativa: {(df_final['label'] == 0).sum()}")

        # Extraer X_sampled (quitando la columna de clase) e y_sampled
        X_sampled = df_final.drop(columns=['label'])
        y_sampled = df_final['label']

        return X_sampled, y_sampled

       
    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        return self.transform(X, y) 