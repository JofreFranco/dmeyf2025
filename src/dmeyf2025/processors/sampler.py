import os
import logging
from typing import Any, Optional
import gc
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample




logger = logging.getLogger(__name__)


class SamplerProcessor():
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

        df_sampled = X
        df_sampled.loc[:, 'label'] = y
        del X, y
        gc.collect()

        # Separar clases
        continua_mask = df_sampled['label'] == 0
        other_classes = df_sampled[~continua_mask]
        continua_cases = df_sampled[continua_mask]
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
        del other_classes, continua_cases, continua_mask, df_sampled
        gc.collect()

        logger.info(f"Dataset final: {len(df_final)} registros")
        logger.info(f"   - Clase positiva: {(df_final['label'] == 1).sum()}")
        logger.info(f"   - Clase negativa: {(df_final['label'] == 0).sum()}")
        
        return df_final.drop(columns=['label']), df_final['label']
