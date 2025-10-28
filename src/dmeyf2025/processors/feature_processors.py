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
    
    def __init__(self, n_lags=1, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label"]):
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
    
    def __init__(self, n_deltas=1, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label"]):
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
    Calcula el ranking percentil de cada cliente para cada variable, agrupado por mes.
    Los percentiles se discretizan en saltos de 5% (0, 5, 10, 15, ..., 95, 100).
    - El valor 0 permanece como 0
    - Los valores positivos se transforman a percentiles discretos
    - Los valores negativos se transforman usando el valor absoluto y luego se aplica el signo negativo
    """
    
    def __init__(self, variables=None, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label"], replace_original=False):
        """
        Parameters:
        -----------
        variables : list, optional
            Lista de variables para calcular percentiles. Si None, usa todas las numéricas
        exclude_cols : list, optional
            Columnas a excluir. Si None, usa EXCLUDE_COLS
        replace_original : bool, default=False
            Si True, reemplaza las columnas originales con los percentiles.
            Si False, crea nuevas columnas con sufijo '_percentile'
        """
        self.variables = variables
        self.exclude_cols = exclude_cols
        self.replace_original = replace_original
        self.selected_variables_ = None
        
    def fit(self, X, y=None):
        """
        Identifica las variables para las cuales calcular percentiles.
        
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
        
        return self
    
    def transform(self, X):
        """
        Aplica la transformación de percentiles discretizados en saltos de 5%.
        Para cada mes y cada variable:
        - El 0 permanece como 0
        - Los valores positivos se rankean entre sí y se discretizan (0, 5, 10, ..., 100)
        - Los valores negativos se rankean usando su valor absoluto, se discretizan y se les aplica signo negativo
        
        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame de entrada
            
        Returns:
        --------
        pd.DataFrame
            Si replace_original=False: DataFrame con nuevas columnas de percentiles (sufijo '_percentile')
            Si replace_original=True: DataFrame con las columnas originales reemplazadas por percentiles
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
            
        X_transformed = X.copy()
        
        # Verificar que existe la columna foto_mes
        if 'foto_mes' not in X_transformed.columns:
            raise ValueError("El DataFrame debe contener la columna 'foto_mes'")
        
        # Aplicar transformación a cada variable seleccionada
        for col in self.selected_variables_:
            if col in X_transformed.columns:
                # Decidir el nombre de la columna de destino
                if self.replace_original:
                    target_col_name = col
                else:
                    target_col_name = f'{col}_percentile'
                
                # Función para calcular percentiles con la lógica especial
                def calculate_percentile_with_sign(group):
                    # Inicializar con NaN
                    result = pd.Series(np.nan, index=group.index)
                    
                    # Los ceros permanecen como 0
                    zero_mask = group == 0
                    result[zero_mask] = 0
                    
                    # Valores positivos
                    pos_mask = group > 0
                    if pos_mask.any():
                        pos_values = group[pos_mask]
                        # Calcular percentil rank (0-100)
                        pos_percentiles = pos_values.rank(pct=True, method='average') * 100
                        # Discretizar en saltos de 5%
                        pos_percentiles_discrete = (pos_percentiles / 5).round() * 5
                        result[pos_mask] = pos_percentiles_discrete
                    
                    # Valores negativos
                    neg_mask = group < 0
                    if neg_mask.any():
                        neg_values = group[neg_mask]
                        # Calcular percentil rank del valor absoluto
                        abs_percentiles = neg_values.abs().rank(pct=True, method='average') * 100
                        # Discretizar en saltos de 5%
                        abs_percentiles_discrete = (abs_percentiles / 5).round() * 5
                        # Aplicar signo negativo
                        result[neg_mask] = -abs_percentiles_discrete
                    
                    return result
                
                # Aplicar la transformación agrupada por foto_mes
                X_transformed[target_col_name] = X_transformed.groupby('foto_mes')[col].transform(
                    calculate_percentile_with_sign
                )
        
        return X_transformed

class DeltaLagTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_deltas=2, n_lags=2, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label"]):
        self.lag_transformer = LagTransformer(n_lags=n_lags, exclude_cols=exclude_cols)
        self.delta_transformer = DeltaTransformer(n_deltas=n_deltas, exclude_cols=exclude_cols)
    
    def fit(self, X, y=None):
        self.lag_transformer.fit(X)
        self.delta_transformer.fit(X)
        return self
    
    def transform(self, X):
        X_transformed = self.lag_transformer.transform(X)
        X_transformed = self.delta_transformer.transform(X_transformed)
        
        return X_transformed

class PeriodStatsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, periods=[12], exclude_cols=["foto_mes", "numero_de_cliente", "target", "label"]):
        self.periods = periods
        self.exclude_cols = exclude_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        numeric_cols = [col for col in X.select_dtypes(include='number').columns if col not in self.exclude_cols]
        X.sort_values(['numero_de_cliente', 'foto_mes'], inplace=True)
        
        new_columns = []
        
        grouped = X.groupby('numero_de_cliente')[numeric_cols]
        
        shifted_data = grouped.shift(1)
        
        for period in self.periods:
            rolling_stats = shifted_data.rolling(window=period, min_periods=1)
            
            for stat_name, stat_func in [('min', 'min'), ('max', 'max'), ('mean', 'mean'), ('median', 'median')]:
                stats_data = getattr(rolling_stats, stat_func)()
                for col in numeric_cols:
                    new_col_name = f'{col}_period{period}_{stat_name}'
                    stats_data[col].name = new_col_name
                    new_columns.append(stats_data[col])
        
        if new_columns:
            X = pd.concat([X] + new_columns, axis=1)
        
        return X

