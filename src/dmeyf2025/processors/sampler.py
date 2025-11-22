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
    
   
    def __init__(
        self, 
        sample_ratio: float = 1, 
        random_state: int = 42, 
        sampling_type: str = "random",
        speed: float = 0.1,
        special_sampling_month = False,
        special_sampling = {}
    ):
        """
        Inicializa el procesador de sampling.
        
        Args:
            sample_ratio: Proporción de muestreo para todas las clases (por defecto 1.0 = sin sampling)
            random_state: Semilla para reproducibilidad
            sampling_type: Tipo de sampling a aplicar: "random", "volta", "linear", "exponential"
            speed: Velocidad de decrecimiento para sampling linear/exponential
                   - Linear: pendiente (ratio = 1 - speed * months_back)
                   - Exponential: factor de decaimiento (ratio = exp(-speed * months_back))
        """
        self.sample_ratio = sample_ratio
        self.sampling_type = sampling_type
        self.speed = speed
        self.debug = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
        self.random_state = random_state
        self.special_sampling = special_sampling
        self.special_sampling_month = special_sampling_month
    def transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        if self.sampling_type == "random":
            return self.random_transform(X, y)
        elif self.sampling_type == "volta":
            return self.volta_transform(X, y)
        elif self.sampling_type == "linear":
            return self.linear_transform(X, y)
        elif self.sampling_type == "exponential":
            return self.exponential_transform(X, y)
        else:
            raise ValueError(f"Sampling type {self.sampling_type} not supported")

    def calculate_equivalent_ratio(self, X: pd.DataFrame, speed: float, sampling_type: str) -> float:
        """
        Mismo objetivo que antes pero ahora como método de instancia:
        - respeta self.special_sampling_month y self.special_sampling
        - usa el mismo orden de meses que los transforms (newest -> oldest)
        """
        if 'foto_mes' not in X.columns:
            raise ValueError("Column 'foto_mes' not found")

        # Asegurar orden consistente: newest -> oldest
        unique_months = sorted(X['foto_mes'].unique(), reverse=True)
        total_records = len(X)
        if total_records == 0:
            return 0.0

        month_counts = X['foto_mes'].value_counts()
        weighted_ratio = 0.0

        for months_back, month in enumerate(unique_months):
            # usa ratio especial si corresponde
            if self.special_sampling_month and (self.special_sampling is not None) and (month in self.special_sampling):
                ratio = float(self.special_sampling[month])
            else:
                if sampling_type == "linear":
                    ratio = 1.0 - speed * months_back
                elif sampling_type == "exponential":
                    # ratio decae desde 1.0 (months_back=0) hacia 0 conforme months_back crece
                    ratio = float(np.exp(-speed * months_back))
                else:
                    raise ValueError(f"Sampling type {sampling_type} not supported")

            # clip a [0,1]
            ratio = max(0.0, min(1.0, ratio))

            month_weight = month_counts.get(month, 0) / total_records
            weighted_ratio += ratio * month_weight

        return weighted_ratio


    def calculate_speed_for_target_ratio(self, X: pd.DataFrame, target_ratio: float, sampling_type: str,
                                         max_iterations: int = 100, tolerance: float = 1e-4,
                                         speed_low: float = 0.0, speed_high: float = 50000.0) -> float:
        """
        Busca el speed que provoque que el ratio equivalente (considerando special_sampling)
        sea aproximadamente target_ratio. Ahora es método de instancia.
        """
        if sampling_type not in ["linear", "exponential"]:
            raise ValueError(f"Sampling type {sampling_type} not supported")

        if not 0.0 < target_ratio <= 1.0:
            raise ValueError(f"target_ratio must be in (0,1], got {target_ratio}")

        best_speed = (speed_low + speed_high) / 2.0
        best_ratio = self.calculate_equivalent_ratio(X, best_speed, sampling_type)

        for it in range(max_iterations):
            mid = (speed_low + speed_high) / 2.0
            current_ratio = self.calculate_equivalent_ratio(X, mid, sampling_type)

            # debug log para seguir el progreso
            logger.debug(f"[calc_speed] it={it} mid={mid:.6f} ratio={current_ratio:.6f}")

            if abs(current_ratio - target_ratio) <= tolerance:
                logger.info(f"Speed encontrado en {it+1} iteraciones: {mid:.6f} -> ratio={current_ratio:.6f}")
                return mid

            # monotonicidad: mayor speed -> menor ratio (para nuestras definiciones)
            if current_ratio > target_ratio:
                # ratio demasiado grande -> aumentar speed -> mover lower bound hacia mid
                speed_low = mid
            else:
                # ratio demasiado chico -> disminuir speed -> mover upper bound hacia mid
                speed_high = mid

            best_speed = mid
            best_ratio = current_ratio

        logger.warning(f"No convergió tras {max_iterations} iteraciones. speed={best_speed:.6f}, ratio={best_ratio:.6f}")
        return best_speed

    def volta_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """
        Volta sampling aplica el sampleo sobre los clientes, manteniendo la historia de los clientes seleccionados
        
        Args:
            X: features
            y: labels
            
        Returns:
            X_sampled, y_sampled: X e y muestreados
        """
        if self.sample_ratio >= 1.0:
            return X, y
        
        if 'numero_de_cliente' not in X.columns:
            raise ValueError("Column 'numero_de_cliente' not found")
        
        unique_clients = X['numero_de_cliente'].unique()
        n_clients_keep = int(len(unique_clients) * self.sample_ratio)
        
        np.random.seed(self.random_state)
        sampled_clients = np.random.choice(
            unique_clients, 
            size=n_clients_keep, 
            replace=False
        )
        
        client_mask = X['numero_de_cliente'].isin(sampled_clients)
        X_sampled = X[client_mask]
        y_sampled = y[client_mask]
        
        logger.info(f"Volta sampling - Dataset final: {len(X_sampled)} registros")
        logger.info(f"   - Clientes originales: {len(unique_clients)}")
        logger.info(f"   - Clientes muestreados: {n_clients_keep}")
        logger.info(f"   - Clase positiva: {(y_sampled == 1).sum()}")
        logger.info(f"   - Clase negativa: {(y_sampled == 0).sum()}")
        logger.info(f"   - Ratio de muestreo equivalente: {len(X_sampled) / len(X)}")
        return X_sampled, y_sampled
    def linear_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """
        Linear sampling aplica sampleo aleatorio con ratio que disminuye linealmente con los meses
        Formula: ratio = 1 - speed * months_back (donde months_back=0 es el mes más reciente)
        
        Args:
            X: features
            y: labels
            
        Returns:
            X_sampled, y_sampled: X e y muestreados
        """
        if 'foto_mes' not in X.columns:
            raise ValueError("Column 'foto_mes' not found")
        
        df_sampled = X.copy()
        df_sampled['label'] = y
        
        unique_months = sorted(df_sampled['foto_mes'].unique(), reverse=True)
        n_months = len(unique_months)
        
        if n_months <= 1:
            return X, y
        
        sampled_dfs = []
        
        for months_back, month in enumerate(unique_months):
            ratio = 1.0 - self.speed * months_back    
            ratio = max(0.0, min(1.0, ratio))  # Clip [0, 1]
            if self.special_sampling_month and month in self.special_sampling.keys():
                ratio = self.special_sampling[month]
            month_data = df_sampled[df_sampled['foto_mes'] == month]
            
            continua_mask = month_data['label'] == 0
            other_classes = month_data[~continua_mask]
            continua_cases = month_data[continua_mask]
            
            n_continua_keep = int(len(continua_cases) * ratio)
            
            if n_continua_keep == 0:
                # Si ratio=0, solo mantener otras clases (positivos)
                month_sampled = other_classes
            elif n_continua_keep >= len(continua_cases):
                # Si ratio≈1, mantener todos
                month_sampled = month_data
            else:
                # Samplear con el ratio calculado
                continua_sampled = resample(
                    continua_cases,
                    n_samples=n_continua_keep,
                    random_state=self.random_state + months_back,
                    replace=False
                )
                month_sampled = pd.concat([other_classes, continua_sampled], ignore_index=True)
            
            if len(month_sampled) > 0:
                sampled_dfs.append(month_sampled)
            
            if self.debug:
                logger.info(f"Linear sampling - Mes {month} (x={months_back}): ratio={ratio:.4f}, registros={len(month_sampled)}")
        
        df_final = pd.concat(sampled_dfs, ignore_index=True)
        
        logger.info(f"Linear sampling - Dataset final: {len(df_final)} registros")
        logger.info(f"   - Clase positiva: {(df_final['label'] == 1).sum()}")
        logger.info(f"   - Clase negativa: {(df_final['label'] == 0).sum()}")
        logger.info(f"   - Ratio de muestreo equivalente: {len(df_final) / len(df_sampled):.4f}")
        
        return df_final.drop(columns=['label']), df_final['label']
    def exponential_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """
        Exponential sampling aplica sampleo aleatorio con ratio que disminuye
        según ratio = exp(-speed * months_back)
        """
    
        if 'foto_mes' not in X.columns:
            raise ValueError("Column 'foto_mes' not found")
    
        # 1) Calcular speed correcto según target_ratio = self.sample_ratio
        speed = self.calculate_speed_for_target_ratio(
            X=X,
            target_ratio=self.sample_ratio,
            sampling_type="exponential"
        )
    
        df_sampled = X.copy()
        df_sampled['label'] = y
    
        # Meses newer -> older
        unique_months = sorted(df_sampled['foto_mes'].unique(), reverse=True)
        if len(unique_months) <= 1:
            return X, y
    
        sampled_dfs = []
    
        for months_back, month in enumerate(unique_months):
    
            # 2) Fórmula correcta
            ratio = float(np.exp(-speed * months_back))
    
            # 3) Override si hay sampling especial
            if self.special_sampling_month and month in self.special_sampling:
                ratio = float(self.special_sampling[month])
    
            # Clampeo
            ratio = max(0.0, min(1.0, ratio))
    
            month_data = df_sampled[df_sampled['foto_mes'] == month]
            continua_mask = month_data['label'] == 0
            other_classes = month_data[~continua_mask]
            continua_cases = month_data[continua_mask]
    
            n_continua_keep = int(len(continua_cases) * ratio)
    
            if n_continua_keep <= 0:
                month_sampled = other_classes
            elif n_continua_keep >= len(continua_cases):
                month_sampled = month_data
            else:
                continua_sampled = resample(
                    continua_cases,
                    n_samples=n_continua_keep,
                    random_state=self.random_state + months_back,
                    replace=False
                )
                month_sampled = pd.concat(
                    [other_classes, continua_sampled],
                    ignore_index=True
                )
    
            sampled_dfs.append(month_sampled)
    
        df_final = pd.concat(sampled_dfs, ignore_index=True)
    
        logger.info(f"Exponential sampling - Dataset final: {len(df_final)} registros")
        logger.info(f"   - Clase positiva: {(df_final['label'] == 1).sum()}")
        logger.info(f"   - Clase negativa: {(df_final['label'] == 0).sum()}")
        logger.info(f"   - Ratio de muestreo equivalente: {len(df_final) / len(df_sampled):.4f}")
    
        return df_final.drop(columns=['label']), df_final['label']

    def random_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
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
