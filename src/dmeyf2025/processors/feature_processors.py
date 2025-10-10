import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from ..utils.data_dict import ALL_CAT_COLS, EXCLUDE_COLS


class LagTransformer(BaseEstimator, TransformerMixin):
    """
    Calcula lags de variables para n_lags meses anteriores.
    Si no hay información del mes anterior, se deja en nulo.
    """
    
    def __init__(self, n_lags=1, exclude_cols=["foto_mes", "numero_de_cliente"]):
        """
        Parameters:
        -----------
        n_lags : int, default=1
            Número de lags a calcular
        exclude_cols : list, optional
            Columnas a excluir del cálculo de lags. Si None, usa EXCLUDE_COLS
        """
        self.n_lags = n_lags
        self.lag_columns_ = None
        self.exclude_cols = exclude_cols
    def fit(self, X, y=None):
        """
        Identifica las columnas para las cuales calcular lags.
        
        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame de entrada
        y : ignored
            No utilizado, presente por compatibilidad con API de sklearn
            
        Returns:
        --------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
            
        # Identificar columnas numéricas para calcular lags (excluir categóricas)
        self.lag_columns_ = [col for col in X.columns 
                            if col not in self.exclude_cols and col not in ALL_CAT_COLS]
        
        return self
    
    def transform(self, X):
        """
        Aplica la transformación de lags.
        
        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame de entrada
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con las nuevas columnas de lags
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
            
        X_transformed = X.copy()
        
        # Preparar lista de nuevas columnas para concatenar
        new_columns = []
        
        # Calcular lags para cada columna
        for lag in range(1, self.n_lags + 1):
            for col in self.lag_columns_:
                if col in X_transformed.columns:
                    # Calcular lag agrupado por cliente
                    lag_col_name = f'{col}_lag{lag}'
                    lag_series = X_transformed.groupby('numero_de_cliente')[col].shift(lag)
                    lag_series.name = lag_col_name
                    new_columns.append(lag_series)
        
        # Concatenar todas las nuevas columnas de una sola vez
        if new_columns:
            X_transformed = pd.concat([X_transformed] + new_columns, axis=1)
        
        return X_transformed


class DeltaTransformer(BaseEstimator, TransformerMixin):
    """
    Calcula deltas de variables respecto al mes anterior.
    Si no hay información del mes anterior, se deja en nulo.
    """
    
    def __init__(self, n_deltas=1, exclude_cols=["foto_mes", "numero_de_cliente"]):
        """
        Parameters:
        -----------
        n_deltas : int, default=1
            Número de deltas a calcular (1 = diferencia con mes anterior, 2 = con 2 meses atrás, etc.)
        exclude_cols : list, optional
            Columnas a excluir del cálculo de deltas. Si None, usa EXCLUDE_COLS
        """
        self.n_deltas = n_deltas
        self.exclude_cols = exclude_cols
        self.delta_columns_ = None
        
    def fit(self, X, y=None):
        """
        Identifica las columnas para las cuales calcular deltas.
        
        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame de entrada
        y : ignored
            No utilizado, presente por compatibilidad con API de sklearn
            
        Returns:
        --------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
            
        # Identificar columnas numéricas para calcular deltas (excluir categóricas)
        self.delta_columns_ = [col for col in X.columns 
                              if col not in self.exclude_cols and col not in ALL_CAT_COLS]
        
        return self
    
    def transform(self, X):
        """
        Aplica la transformación de deltas.
        
        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame de entrada
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con las nuevas columnas de deltas
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
            
        X_transformed = X.copy()
        
        # Preparar lista de nuevas columnas para concatenar
        new_columns = []
        
        # Calcular deltas para cada columna
        for delta in range(1, self.n_deltas + 1):
            for col in self.delta_columns_:
                if col in X_transformed.columns:
                    # Calcular delta agrupado por cliente
                    delta_col_name = f'{col}_delta{delta}'
                    lagged_col = X_transformed.groupby('numero_de_cliente')[col].shift(delta)
                    delta_series = X_transformed[col] - lagged_col
                    delta_series.name = delta_col_name
                    new_columns.append(delta_series)
        
        # Concatenar todas las nuevas columnas de una sola vez
        if new_columns:
            X_transformed = pd.concat([X_transformed] + new_columns, axis=1)
        
        return X_transformed


class PercentileTransformer(BaseEstimator, TransformerMixin):
    """
    Calcula percentiles de variables seleccionadas y crea nuevas features basadas en estos percentiles.
    """
    
    def __init__(self, variables=None, percentiles=[25, 50, 75, 90, 95], exclude_cols=["foto_mes", "numero_de_cliente"]):
        """
        Parameters:
        -----------
        variables : list, optional
            Lista de variables para calcular percentiles. Si None, usa todas las numéricas
        percentiles : list, default=[25, 50, 75, 90, 95]
            Lista de percentiles a calcular
        exclude_cols : list, optional
            Columnas a excluir. Si None, usa EXCLUDE_COLS
        """
        self.variables = variables
        self.percentiles = percentiles
        self.exclude_cols = exclude_cols
        self.percentile_values_ = {}
        self.selected_variables_ = None
        
    def fit(self, X, y=None):
        """
        Calcula los percentiles para las variables seleccionadas.
        
        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame de entrada
        y : ignored
            No utilizado, presente por compatibilidad con API de sklearn
            
        Returns:
        --------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
            
        # Seleccionar variables si no se especificaron
        if self.variables is None:
            self.selected_variables_ = [col for col in X.columns 
                                      if col not in self.exclude_cols and col not in ALL_CAT_COLS
                                      and X[col].dtype in ['int64', 'float64']]
        else:
            self.selected_variables_ = [col for col in self.variables if col in X.columns]
        
        # Calcular percentiles para cada variable
        for col in self.selected_variables_:
            self.percentile_values_[col] = {}
            for percentile in self.percentiles:
                self.percentile_values_[col][percentile] = np.percentile(X[col].dropna(), percentile)
        
        return self
    
    def transform(self, X):
        """
        Aplica la transformación de percentiles.
        
        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame de entrada
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con las nuevas columnas de percentiles
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
            
        X_transformed = X.copy()
        
        # Preparar lista de nuevas columnas para concatenar
        new_columns = []
        
        # Crear features basadas en percentiles
        for col in self.selected_variables_:
            if col in X_transformed.columns:
                for percentile in self.percentiles:
                    threshold = self.percentile_values_[col][percentile]
                    
                    # Crear columna binaria: 1 si está por encima del percentile, 0 si no
                    percentile_col_name = f'{col}_above_p{percentile}'
                    above_series = (X_transformed[col] > threshold).astype(int)
                    above_series.name = percentile_col_name
                    new_columns.append(above_series)
                    
                    # Crear columna con la distancia al percentile
                    distance_col_name = f'{col}_dist_p{percentile}'
                    distance_series = X_transformed[col] - threshold
                    distance_series.name = distance_col_name
                    new_columns.append(distance_series)
        
        # Concatenar todas las nuevas columnas de una sola vez
        if new_columns:
            X_transformed = pd.concat([X_transformed] + new_columns, axis=1)
        
        return X_transformed

class DeltaLagTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_deltas=2, n_lags=2):
        self.lag_transformer = LagTransformer(n_lags=n_lags)
        self.delta_transformer = DeltaTransformer(n_deltas=n_deltas)
    
    def fit(self, X, y=None):
        self.lag_transformer.fit(X)
        self.delta_transformer.fit(X)
        return self
    
    def transform(self, X):
        X_transformed = self.lag_transformer.transform(X)
        X_transformed = self.delta_transformer.transform(X_transformed)
        
        return X_transformed