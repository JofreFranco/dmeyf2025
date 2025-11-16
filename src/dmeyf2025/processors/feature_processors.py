import pandas as pd
from scipy import stats
import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from flaml.default import preprocess_and_suggest_hyperparams

from ..utils.data_dict import ALL_CAT_COLS, EXCLUDE_COLS

import logging
logger = logging.getLogger(__name__)
# Función para calcular percentiles con la lógica especial
def calculate_percentile_with_sign(group, n_bins=None):
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
        # Discretizar en saltos de n_bins
        if n_bins is not None:
            pos_percentiles_discrete = (pos_percentiles / n_bins).round() * n_bins
            result[pos_mask] = pos_percentiles_discrete
        else:
            result[pos_mask] = pos_percentiles
    return result
class BaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):

        if "clase_ternaria" in X.columns:
            raise ValueError("La columna 'clase_ternaria' no debe estar en el dataset")
        X_transformed = self._transform(X)
        return X_transformed
class CleanZerosTransformer(BaseTransformer):

    """
    Detecta pares de variables cVARIABLE (cantidad) y mVARIABLE (monto).
    Cuando la cantidad es 0, pone el monto en None en lugar de 0.
    
    Esto evita que el modelo aprenda relaciones incorrectas donde hay montos
    en 0 que en realidad deberían ser valores nulos (porque no existe la operación).

    Además limpia el mes 202006 que tiene muchas variables rotas en 0, se pasan a Nan y también se limpia la variable de mobile app que se invirtió, poniendo antes de la inversión todo en Nan
    """
    def __init__(self, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"]):
        self.exclude_cols = exclude_cols
        self.variable_pairs_ = None
    
    def fit(self, X, y=None):
        """Identifica los pares de variables cVARIABLE y mVARIABLE."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
        
        # Detectar todas las columnas que empiezan con 'c' y 'm'
        c_columns = [col for col in X.columns if col.startswith('c') and col not in self.exclude_cols]
        m_columns = [col for col in X.columns if col.startswith('m') and col not in self.exclude_cols]
        
        # Encontrar pares donde existe tanto cVARIABLE como mVARIABLE
        self.variable_pairs_ = []
        for c_col in c_columns:
            # Extraer el nombre de la variable (sin el prefijo 'c')
            var_name = c_col[1:]  # Eliminar la 'c' del inicio
            m_col = 'm' + var_name
            
            if m_col in m_columns:
                self.variable_pairs_.append((c_col, m_col))
        
        return self
    
    def _transform(self, X):
        """
        Pone en None los montos (mVARIABLE) cuando la cantidad (cVARIABLE) es 0.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
        
        X_transformed = X
        
        # Para cada par detectado, limpiar los montos cuando cantidad = 0
        for c_col, m_col in self.variable_pairs_:
            if c_col in X_transformed.columns and m_col in X_transformed.columns:
                # Crear máscara donde la cantidad es 0
                zero_mask = X_transformed[c_col] == 0
                
                # Poner el monto en None donde la cantidad es 0
                X_transformed.loc[zero_mask, m_col] = np.nan
        
        # Limpiar antes del mes 202010 tmobile_app
        X_transformed.loc[X_transformed["foto_mes"] < 202010, "tmobile_app"] = np.nan

        # Limpiar variables rotas del 202006
        #detectar todas las columnas que son SOLO 0 en el 202006
        zero_cols = X_transformed[X_transformed["foto_mes"] == 202006].columns[X_transformed[X_transformed["foto_mes"] == 202006].apply(lambda x: x.nunique()) == 1].drop("foto_mes")
        
        X_transformed.loc[X_transformed["foto_mes"] == 202006, zero_cols] = np.nan        
        return X_transformed



class PercentileTransformer(BaseTransformer):
    """
    Calcula el ranking percentil de cada cliente para cada variable, agrupado por mes.
    - El valor 0 permanece como 0
    - Los valores positivos se transforman a percentiles (0-100)
    - Los valores negativos se transforman usando el valor absoluto y luego se aplica el signo negativo
    """
    
    def __init__(self, variables=None, exclude_cols=None, replace_original=True):
        """
        Aplica la transformación de percentiles continuos.
        Para cada mes y cada variable:
        - El 0 permanece como 0
        - Los valores positivos se rankean entre sí (0-100)
        - Los valores negativos se rankean usando su valor absoluto y se les aplica signo negativo (-100 a 0)
        
        Parameters:
        -----------
        variables : list, optional
            Lista de variables para calcular percentiles. Si None, usa todas las numéricas
        exclude_cols : list, optional
            Columnas a excluir. Si None, usa ["foto_mes", "numero_de_cliente", "target", "label", "weight"]
        replace_original : bool, default=False
            Si True, reemplaza las columnas originales con los percentiles.
            Si False, crea nuevas columnas con sufijo '_percentile'
        """
        self.variables = variables
        self.exclude_cols = exclude_cols if exclude_cols is not None else ["foto_mes", "numero_de_cliente", "target", "label", "weight"]
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
        
        # Verificar que existe la columna foto_mes
        if 'foto_mes' not in X.columns:
            raise ValueError("El DataFrame debe contener la columna 'foto_mes'")
            
        # Seleccionar variables si no se especificaron
        if self.variables is None:
            self.selected_variables_ = [
                col for col in X.columns 
                if col not in self.exclude_cols]
        else:
            # Verificar que todas las variables existen
            missing_cols = set(self.variables) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Las siguientes variables no existen en X: {missing_cols}")
            self.selected_variables_ = list(self.variables)
        
        return self
    
    def _calculate_percentile(self, values):
        """
        Calcula percentiles continuos con manejo de ceros y negativos.
        Parameters:
        -----------
        values : np.ndarray
            Array de valores a transformar
            
        Returns:
        --------
        np.ndarray
            Array de percentiles (float32)
        """
        n = len(values)
        if n == 0:
            return np.array([], dtype=np.float32)
        
        # Inicializar resultado con ceros
        result = np.zeros(n, dtype=np.float32)        
        # Separar positivos y negativos, los 0 quedan como estan
        pos_mask = values > 0
        neg_mask = values < 0
        
        # Procesar valores positivos
        if pos_mask.any():
            pos_values = values[pos_mask]
            pos_ranks = stats.rankdata(pos_values, method='min')
            # Calcular percentiles (0-100)
            if len(pos_values) > 1:
                pos_percentiles = (pos_ranks - 1) / (len(pos_values) - 1) * 100
            else:
                pos_percentiles = np.array([50.0])
            result[pos_mask] = pos_percentiles.astype(np.float32)
        
        # Procesar valores negativos
        if neg_mask.any():
            neg_values = np.abs(values[neg_mask])
            neg_ranks = stats.rankdata(neg_values, method='min')
            # Calcular percentiles y aplicar signo negativo
            if len(neg_values) > 1:
                neg_percentiles = (neg_ranks - 1) / (len(neg_values) - 1) * 100
            else:
                neg_percentiles = np.array([50.0])
            result[neg_mask] = -neg_percentiles.astype(np.float32)
        
        return result
    
    
    def _transform(self, X):
        
        def calculate_percentile_for_group(series):
            """
            Calcula percentiles para un mes específico.
            - 0 permanece como 0
            - Positivos se rankean entre sí (0-100)
            - Negativos se rankean usando valor absoluto y se aplica signo negativo (-100 a 0)
            """
            v = series.values
            
            # Máscaras separadas
            mask_nan = np.isnan(v)
            mask_zero = (v == 0)
            mask_pos = (v > 0)
            mask_neg = (v < 0)
            
            # Inicializar resultado
            result = np.zeros_like(v, dtype=np.float32)
            
            # === Positivos ===
            if mask_pos.any():
                pos_values = v[mask_pos]
                
                # rank sin nans
                ranks = stats.rankdata(pos_values, method='min')
                if len(pos_values) > 1:
                    percentiles = (ranks - 1) / (len(pos_values) - 1) * 100
                else:
                    percentiles = np.array([50.0])
                
                result[mask_pos] = percentiles.astype(np.float32)
            
            # === Negativos ===
            if mask_neg.any():
                neg_values = np.abs(v[mask_neg])
                
                ranks = stats.rankdata(neg_values, method='min')
                if len(neg_values) > 1:
                    percentiles = (ranks - 1) / (len(neg_values) - 1) * 100
                else:
                    percentiles = np.array([50.0])
                
                result[mask_neg] = -percentiles.astype(np.float32)
            
            # === Restituir ceros ===
            result[mask_zero] = 0.0
            
            # === Restituir NaNs ===
            result[mask_nan] = np.nan
            
            return pd.Series(result, index=series.index)
        
        # Procesar cada columna, agrupando por foto_mes
        for col in self.selected_variables_:
            # Agrupar por foto_mes y aplicar la transformación de percentiles
            X[col] = X.groupby('foto_mes')[col].transform(calculate_percentile_for_group)
        
        return X
    def fit_transform(self, X, y=None):
        """Ajusta y transforma en un solo paso"""
        return self.fit(X, y).transform(X)

class LagTransformer(BaseTransformer):
    """
    Calcula lags de variables para n_lags meses anteriores.
    Si no hay información del mes anterior, se deja en nulo.
    """
    
    def __init__(self, n_lags=1, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"]):
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
    
    def _transform(self, X):
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
            
        X_transformed = X
        
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

class DeltaTransformer(BaseTransformer):
    """
    Calcula deltas de variables respecto al mes anterior.
    Si no hay información del mes anterior, se deja en nulo.
    """
    
    def __init__(self, n_deltas=1, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"]):
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
    
    def _transform(self, X_transformed):
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
        if not isinstance(X_transformed, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")

        
        # Preparar lista de nuevas columnas para concatenar
        new_columns = []
        
        # Calcular deltas para cada columna
        for delta in range(1, self.n_deltas + 1):
            for col in self.delta_columns_:
                if col in X_transformed.columns:
                    # Calcular delta agrupado por cliente
                    delta_col_name = f'{col}_delta{delta}'
                    lag_col_name = f"'{col}lag{delta}'"
                    lagged_col = X_transformed.groupby('numero_de_cliente')[col].shift(delta)
                    lagged_col.name = lag_col_name
                    delta_series = X_transformed[col] - lagged_col
                    delta_series.name = delta_col_name
                    new_columns.append(delta_series)
                    new_columns.append(lagged_col)
        
        # Concatenar todas las nuevas columnas de una sola vez
        if new_columns:
            X_transformed = pd.concat([X_transformed] + new_columns, axis=1)
        
        return X_transformed

class DeltaLagTransformer(BaseTransformer):
    """
    Calcula lags y deltas de variables de forma optimizada.
    Para cada columna genera: col_lag1, col_lag2, col_delta1, col_delta2
    """
    
    def __init__(self, n_lags=2, exclude_cols=None):
        """
        Parameters:
        -----------
        n_lags : int, default=2
            Número de lags y deltas a calcular
        exclude_cols : list, optional
            Columnas a excluir del cálculo
        """
        self.n_lags = n_lags
        self.exclude_cols = exclude_cols if exclude_cols is not None else [
            "foto_mes", "numero_de_cliente", "target", "label", "weight"
        ]
        self.feature_columns_ = None
    
    def fit(self, X, y=None):
        """Identifica las columnas para procesar"""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
        
        if "clase_ternaria" in X.columns:
            raise ValueError("La columna 'clase_ternaria' no debe estar en el dataset")
        
        # Verificar columnas requeridas
        if 'numero_de_cliente' not in X.columns:
            raise ValueError("El DataFrame debe contener 'numero_de_cliente'")
        
        # Identificar columnas a procesar
        self.feature_columns_ = [
            col for col in X.columns 
            if col not in self.exclude_cols and col not in ALL_CAT_COLS
        ]
        
        return self
    
    def _transform(self, X):
        """
        Aplica transformación optimizada de lags y deltas.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
        
        if self.feature_columns_ is None:
            raise ValueError("El transformador no ha sido ajustado. Llame a fit() primero.")
        
        # Ordenar por cliente (asegura que shift funcione correctamente)
        # Pandas groupby.shift ya maneja esto internamente, pero lo hacemos explícito
        X_sorted = X.sort_values(['numero_de_cliente', 'foto_mes'])
        
        # Obtener solo las columnas a procesar
        cols_to_process = [col for col in self.feature_columns_ if col in X_sorted.columns]
        
        if not cols_to_process:
            return X
        
        # Crear un DataFrame con las columnas a procesar
        data_to_shift = X_sorted[cols_to_process]
        
        # Agrupar una sola vez
        grouped = X_sorted.groupby('numero_de_cliente', sort=False)
        
        # Diccionario para almacenar todas las nuevas columnas
        new_cols_dict = {}
        
        # Calcular todos los lags de una sola vez
        for lag in range(1, self.n_lags + 1):
            # Shift de todas las columnas a la vez
            lagged_data = grouped[cols_to_process].shift(lag)
            
            # Renombrar columnas para lags
            lagged_data.columns = [f'{col}_lag{lag}' for col in cols_to_process]
            
            # Agregar al diccionario
            for col_name in lagged_data.columns:
                new_cols_dict[col_name] = lagged_data[col_name].values
            
            # Calcular deltas usando los lags recién calculados
            # delta = valor_actual - valor_lag
            for i, col in enumerate(cols_to_process):
                delta_col_name = f'{col}_delta{lag}'
                # Usar .values para operación numpy (más rápido)
                new_cols_dict[delta_col_name] = data_to_shift[col].values - lagged_data.iloc[:, i].values
        
        # Crear DataFrame con todas las nuevas columnas de una sola vez
        new_cols_df = pd.DataFrame(new_cols_dict, index=X_sorted.index)
        
        # Concatenar una sola vez
        X_transformed = pd.concat([X_sorted, new_cols_df], axis=1)
        
        # Restaurar el orden original si es necesario
        X_transformed = X_transformed.loc[X.index]
        
        return X_transformed
    
    def transform(self, X):
        """Alias para _transform"""
        return self._transform(X)
    
    def fit_transform(self, X, y=None):
        """Ajusta y transforma en un solo paso"""
        return self.fit(X, y).transform(X)

class PeriodStatsTransformer(BaseTransformer):
    def __init__(self, periods=[6], exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"]):
        self.periods = periods
        self.exclude_cols = exclude_cols
    
    def fit(self, X, y=None):
        return self
    
    def _transform(self, X):

        numeric_cols = [col for col in X.select_dtypes(include='number').columns if col not in self.exclude_cols and col.startswith('m')]
        X.sort_values(['numero_de_cliente', 'foto_mes'], inplace=True)
        
        new_columns = []
        
        grouped = X.groupby('numero_de_cliente')[numeric_cols]
        
        shifted_data = grouped.shift(1)
        
        for period in self.periods:
            rolling_stats = shifted_data.rolling(window=period, min_periods=1)
            
            for stat_name, stat_func in [('min', 'min'), ('max', 'max'), ('mean', 'mean'), ('std', 'std')]:
                stats_data = getattr(rolling_stats, stat_func)()
                for col in numeric_cols:
                    new_col_name = f'{col}_period{period}_{stat_name}'
                    stats_data[col].name = new_col_name
                    new_columns.append(stats_data[col])
        
        if new_columns:
            X = pd.concat([X] + new_columns, axis=1)
        
        return X

class LegacyTendencyTransformer(BaseTransformer):
    """
    Calcula la pendiente de regresión lineal de cada variable numérica para cada cliente.
    Usa una expanding window: para cada mes, calcula la tendencia usando todos los datos históricos.
    """
    def __init__(self, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"]):
        self.exclude_cols = exclude_cols
        self.numeric_cols_ = None
    
    def fit(self, X, y=None):
        """Identifica las columnas numéricas para calcular tendencias."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
        
        self.numeric_cols_ = [col for col in X.columns 
                             if col not in self.exclude_cols and col not in ALL_CAT_COLS and col.startswith('m')]
        return self
    
    def _transform(self, X):
        """Calcula la pendiente de regresión lineal para cada variable y cliente."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
        
        X_transformed = X
        X_transformed = X_transformed.sort_values(['numero_de_cliente', 'foto_mes'])
        
        # Función para calcular pendiente usando fórmula de mínimos cuadrados
        def calculate_slope(series):
            valid_mask = series.notna()
            if valid_mask.sum() < 2:
                return np.nan
            
            y = series[valid_mask].values
            x = np.arange(len(y))
            
            # Fórmula de mínimos cuadrados: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
            n = len(x)
            sum_x = x.sum()
            sum_y = y.sum()
            sum_xy = (x * y).sum()
            sum_x2 = (x * x).sum()
            
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope
        
        # Calcular pendiente para cada variable numérica
        new_columns = []
        for col in self.numeric_cols_:
            if col in X_transformed.columns:
                # Usar expanding para calcular la pendiente con todos los datos históricos
                slope_col = X_transformed.groupby('numero_de_cliente')[col].expanding().apply(
                    calculate_slope, raw=False
                ).reset_index(level=0, drop=True)
                
                slope_col.name = f'{col}_tendency'
                new_columns.append(slope_col)

        # Concatenar todas las nuevas columnas
        if new_columns:
            X_transformed = pd.concat([X_transformed] + new_columns, axis=1)
        
        return X_transformed

class TendencyTransformer(BaseTransformer):
    """
    Calcula la pendiente de regresión lineal de cada variable numérica para cada cliente usando una ventana de 6 meses.
    """
    def __init__(self, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"], window=6):
        self.exclude_cols = exclude_cols
        self.numeric_cols_ = None
        self.window = window

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")

        self.numeric_cols_ = [
            col for col in X.columns
            if col not in self.exclude_cols and col not in ALL_CAT_COLS and col.startswith('m')
        ]
        return self

    def _transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")

        X = X.sort_values(['numero_de_cliente', 'foto_mes'])
        clientes = X['numero_de_cliente'].values
        new_cols = {}

        # identificar cortes por cliente (para procesar cada bloque como un array contiguo)
        _, start_idx, counts = np.unique(clientes, return_index=True, return_counts=True)

        for col in self.numeric_cols_:
            y_all = X[col].values.astype(float)
            slope = np.full_like(y_all, np.nan, dtype=float)

            for s, n in zip(start_idx, counts):
                y = y_all[s : s + n]
                mask = np.isfinite(y)
                
                # Para cada posición, calcular la pendiente con los últimos window meses
                for i in range(n):
                    # Determinar la ventana: últimos window meses (incluyendo el actual)
                    start_window = max(0, i - self.window + 1)
                    end_window = i + 1
                    
                    # Extraer datos de la ventana
                    y_window = y[start_window:end_window]
                    mask_window = mask[start_window:end_window]
                    
                    # Verificar que hay al menos 2 datos válidos
                    if mask_window.sum() < 2:
                        continue
                    
                    # Filtrar solo valores válidos
                    y_valid = y_window[mask_window]
                    x_valid = np.arange(len(y_window))[mask_window].astype(float)
                    
                    # Calcular pendiente con fórmula de mínimos cuadrados
                    n_valid = len(x_valid)
                    sum_x = x_valid.sum()
                    sum_y = y_valid.sum()
                    sum_xy = (x_valid * y_valid).sum()
                    sum_x2 = (x_valid * x_valid).sum()
                    
                    den = n_valid * sum_x2 - sum_x * sum_x
                    if den != 0:
                        num = n_valid * sum_xy - sum_x * sum_y
                        slope[s + i] = num / den

            new_cols[f"{col}_tendency"] = slope

        # concatenar todo en el DataFrame final
        X_out = X.assign(**new_cols)
        return X_out

class IntraMonthTransformer(BaseTransformer):

    def __init__(self, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"]):
        self.exclude_cols = exclude_cols
    
    def fit(self, X, y=None):
        return self
    
    def _transform(self, X):
        X_transformed = X
        # Ratios
        X_transformed["ratio_rentabilidad_activos"] = X_transformed["mrentabilidad_annual"] / (X_transformed["mactivos_margen"] + np.finfo(float).eps)
        X_transformed["ratio_rentabilidad_pasivos"] = X_transformed["mrentabilidad_annual"] / (X_transformed["mpasivos_margen"] + np.finfo(float).eps)
        X_transformed["ratio_liquidez_relativa"] = X_transformed["mcuentas_saldo"] / (X_transformed["mpasivos_margen"] + np.finfo(float).eps)
        
        X_transformed["ratio_intensidad_credito"] = (X_transformed["mtarjeta_visa_consumo"] + X_transformed["mtarjeta_master_consumo"]) / (X_transformed["Visa_mlimitecompra"] + X_transformed["Master_mlimitecompra"] + np.finfo(float).eps)
    
        # Totals
        X_transformed["total_limite_credito"] = X_transformed["Visa_mlimitecompra"] + X_transformed["Master_mlimitecompra"]

        X_transformed["total_consumo_tarjetas"] = X_transformed["mtarjeta_visa_consumo"] + X_transformed["mtarjeta_master_consumo"]
    
        # Ratios adicionales
        X_transformed["cproductos_por_antiguedad"] = X_transformed["cproductos"] / (X_transformed["cliente_antiguedad"] + np.finfo(float).eps)
        X_transformed["comision_por_producto"] = X_transformed["mcomisiones"] / (X_transformed["cproductos"] + np.finfo(float).eps)
        
        # Sumas de variables relacionadas
        X_transformed["deliquency"] = (X_transformed["Visa_delinquency"] + X_transformed["Master_delinquency"])
        X_transformed["status"] = (X_transformed["Visa_status"] + X_transformed["Master_status"])
        
        return X_transformed

class HistoricalFeaturesTransformer(BaseTransformer):
    """
    Transformer que genera features históricos basados en datos de meses anteriores.
    Incluye promedios móviles, comparaciones temporales y eventos históricos.
    """
    def __init__(self, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"]):
        self.exclude_cols = exclude_cols

    def fit(self, X, y=None):
        if "clase_ternaria" in X.columns:
            raise ValueError("La columna 'clase_ternaria' no debe estar en el dataset")
        return self

    def _transform(self, X):
        X_transformed = X
        
        # Promedios móviles de transacciones con tarjetas
        X_transformed = X_transformed.sort_values(["numero_de_cliente", "foto_mes"])
        X_transformed["visa_txn_cnt_3m_avg"] = (
            X_transformed.groupby("numero_de_cliente")["ctarjeta_visa_transacciones"]
            .shift(1)
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        X_transformed["visa_txn_cnt_3m_avg_ratio"] = X_transformed["ctarjeta_visa_transacciones"] / (X_transformed["visa_txn_cnt_3m_avg"] + np.finfo(float).eps)

        X_transformed = X_transformed.sort_values(["numero_de_cliente", "foto_mes"])
        X_transformed["master_txn_cnt_3m_avg"] = (
            X_transformed.groupby("numero_de_cliente")["ctarjeta_master_transacciones"]
            .shift(1)
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        X_transformed["master_txn_cnt_3m_avg_ratio"] = X_transformed["ctarjeta_master_transacciones"] / (X_transformed["master_txn_cnt_3m_avg"] + np.finfo(float).eps)

        # Meses desde eventos relevantes
        X_transformed["meses_desde_ult_pago_servicio"] = (
            X_transformed
            .sort_values(["numero_de_cliente", "foto_mes"])
            .groupby("numero_de_cliente")["mpagodeservicios"]
            .apply(lambda s: (~(s > 0)).cumsum() - (~(s > 0)).cumsum().where(s > 0).ffill().fillna(0).astype(int))
            .reset_index(level=0, drop=True)
        )
        X_transformed["meses_desde_ult_pagomiscuentas"] = (
            X_transformed
            .sort_values(["numero_de_cliente", "foto_mes"])
            .groupby("numero_de_cliente")["mpagomiscuentas"]
            .apply(lambda s: (~(s > 0)).cumsum() - (~(s > 0)).cumsum().where(s > 0).ffill().fillna(0).astype(int))
            .reset_index(level=0, drop=True)
        )

        # Comparación con mes anterior
        X_transformed = X_transformed.sort_values(["numero_de_cliente", "foto_mes"])
        X_transformed["cproductos_anterior"] = (
            X_transformed.groupby("numero_de_cliente")["cproductos"].shift(1)
        )
        X_transformed["perdio_producto_1m"] = (
            (X_transformed["cproductos"] < X_transformed["cproductos_anterior"]).astype(int)
        )
        X_transformed = X_transformed.drop(columns=["cproductos_anterior"])

        # Fricción (basado en promedio histórico)
        X_transformed["ccajas_consultas_3m_avg"] = (
            X_transformed
                .sort_values(["numero_de_cliente", "foto_mes"])
                .groupby("numero_de_cliente")["ccajas_consultas"]
                .shift(1)
                .rolling(window=3, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
        )
        X_transformed["friccion"] = X_transformed["ccajas_consultas"] / (X_transformed["ccajas_consultas_3m_avg"] + 1)
        
        return X_transformed

class DatesTransformer(BaseTransformer):
    def __init__(self, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"]):
        self.exclude_cols = exclude_cols

    def fit(self, X, y=None):
        return self
    
    def _transform(self, X):
        X_transformed = X
        X_transformed["Master_fechaalta_meses"] = X_transformed["Master_fechaalta"].apply(lambda x: int(x / 28) if pd.notnull(x) else None)
        X_transformed["Visa_fechaalta_meses"] = X_transformed["Visa_fechaalta"].apply(lambda x: int(x / 28) if pd.notnull(x) else None)
        X_transformed["Master_recencia"] = X_transformed["cliente_antiguedad"] - X_transformed["Master_fechaalta_meses"]
        X_transformed["Visa_recencia"] = X_transformed["cliente_antiguedad"] - X_transformed["Visa_fechaalta_meses"]

        return X_transformed

class RandomForestFeaturesTransformer(BaseTransformer):
    def __init__(self, exclude_cols=["numero_de_cliente", "label", "weight", "clase_ternaria", "target"], n_estimators=20, num_leaves=16, min_data_in_leaf=100, feature_fraction_bynode=0.2, training_months= [], use_zero_shot=False):
        self.exclude_cols = exclude_cols
        self.training_months = training_months  
        self.n_estimators = n_estimators
        self.use_zero_shot = use_zero_shot
        self.lgb_params = {
            "num_iterations": n_estimators,
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf,
            "feature_fraction_bynode": feature_fraction_bynode,
            "boosting": "rf",
            "bagging_fraction": (1.0 - 1.0 / np.exp(1.0)),
            "bagging_freq": 1,
            "feature_fraction": 1.0,
            "max_bin": 31,
            "objective": "binary",
            "first_metric_only": True,
            "boost_from_average": True,
            "feature_pre_filter": False,
            "force_row_wise": True,
            "verbosity": -100,
            "max_depth": -1,
            "min_gain_to_split": 0.0,
            "min_sum_hessian_in_leaf": 0.001,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "pos_bagging_fraction": 1.0,
            "neg_bagging_fraction": 1.0,
            "is_unbalance": False,
            "scale_pos_weight": 1.0,
            "drop_rate": 0.1,
            "max_drop": 50,
            "skip_drop": 0.5,
            "extra_trees": False
    }
    
    def fit(self, X, y=None):
        logger.info(f"Entrenando RandomForestFeaturesTransformer con {self.training_months} meses")
        X = X
        X_train = X.loc[X["foto_mes"].isin(self.training_months)]
        y = X_train["label"]
        self.keep_cols = [col for col in X_train.columns if col not in self.exclude_cols]
        X_train = X_train[self.keep_cols]
        self.columns_ = X_train.columns
        
        if self.use_zero_shot:
            (
            hp,
            estimator_class,
            X_transformed,
            y_transformed,
            feature_transformer,
            label_transformer,
            ) = preprocess_and_suggest_hyperparams("classification", X, y, "rf")
            self.lgb_params.update({"num_iterations": hp["n_estimators"], "num_leaves": hp["max_leaf_nodes"], "feature_fraction": hp["max_features"]})
        dtrain = lgb.Dataset(
        data=X_train.values,
        label=y,
        free_raw_data=False)
        self.model_ = lgb.train(params=self.lgb_params, train_set=dtrain)
        logger.info("RandomForestFeaturesTransformer entrenado")
        
        return self
    
    def _transform(self, X):
        X = X
        extra_cols = set(X.columns) - set(self.keep_cols)
        extra_cols = X[list(extra_cols)]
        X = X[self.keep_cols]

        prediccion = self.model_.predict(X.values, pred_leaf=True)
        prediccion = np.array(prediccion, dtype=int)

        n_obs, n_trees = prediccion.shape
        logger.info(f"Generando {n_trees} árboles de features...")
        new_cols = {}
        for tree in range(n_trees):
            leaves = np.unique(prediccion[:, tree])
            for leaf in leaves:
                varname = f"rf_{tree + 1:03d}_{leaf:03d}"
                new_cols[varname] = (prediccion[:, tree] == leaf).astype(int)
        
        if new_cols:
            logger.info(f"Se generaron {len(new_cols)} nuevas columnas")
            new_cols_df = pd.DataFrame(new_cols, index=X.index)
            X = pd.concat([X, new_cols_df, extra_cols], axis=1)
        else:
            logger.info("No se generaron nuevas columnas")
            X = pd.concat([X, extra_cols], axis=1)
        
        return X
class AddCanaritos(BaseTransformer):
    def __init__(self, n_canaritos=10):
        self.n_canaritos = n_canaritos

    def fit(self, X, y=None):
        return self

    def _transform(self, X):
        X_transformed = X
        other_cols = list(X_transformed.columns)
        canarito_cols = []
        new_columns = {}
        for i in range(self.n_canaritos):
            col_name = f"canarito_{i}"
            new_columns[col_name] = np.random.rand(len(X_transformed))
            canarito_cols.append(col_name)
        # Crear un solo diccionario con todas las columnas
        full_columns = {}
        # Primero los canaritos
        for col_name in canarito_cols:
            full_columns[col_name] = new_columns[col_name]
        # Luego las columnas originales
        for col_name in other_cols:
            full_columns[col_name] = X_transformed[col_name]
        # Consolidar dataframe de una sola vez para evitar fragmentación
        X_transformed = pd.DataFrame(full_columns, index=X_transformed.index)
        return X_transformed