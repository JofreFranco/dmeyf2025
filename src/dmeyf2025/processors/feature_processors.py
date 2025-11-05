import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from ..utils.data_dict import ALL_CAT_COLS, EXCLUDE_COLS

import logging
logger = logging.getLogger(__name__)

class LagTransformer(BaseEstimator, TransformerMixin):
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
    
    def __init__(self, variables=None, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"], replace_original=False):
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
                                      and col.startswith('m')]
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
    def __init__(self, n_deltas=2, n_lags=2, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"]):
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
    def __init__(self, periods=[12], exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"]):
        self.periods = periods
        self.exclude_cols = exclude_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        numeric_cols = [col for col in X.select_dtypes(include='number').columns if col not in self.exclude_cols and col.startswith('m')]
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

class LegacyTendencyTransformer(BaseEstimator, TransformerMixin):
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
    
    def transform(self, X):
        """Calcula la pendiente de regresión lineal para cada variable y cliente."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
        
        X_transformed = X.copy()
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

class TendencyTransformer(BaseEstimator, TransformerMixin):
    """
    Calcula la pendiente de regresión lineal de cada variable numérica para cada cliente.
    Usa una ventana expanding: para cada mes, calcula la tendencia usando todos los datos históricos.
    Implementación vectorizada con NumPy (sin apply ni loops internos lentos).
    """
    def __init__(self, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"]):
        self.exclude_cols = exclude_cols
        self.numeric_cols_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")

        self.numeric_cols_ = [
            col for col in X.columns
            if col not in self.exclude_cols and col not in ALL_CAT_COLS and col.startswith('m')
        ]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")

        X = X.sort_values(['numero_de_cliente', 'foto_mes']).copy()
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
                if mask.sum() < 2:
                    continue

                y = np.where(mask, y, 0.0)  # reemplazo NaN temporal
                x = np.arange(n, dtype=float)

                # acumulativas
                n_valid = np.cumsum(mask)
                sum_x = np.cumsum(x * mask)
                sum_y = np.cumsum(y)
                sum_xy = np.cumsum(x * y)
                sum_x2 = np.cumsum(x * x)

                # fórmula de la pendiente
                num = n_valid * sum_xy - sum_x * sum_y
                den = n_valid * sum_x2 - sum_x * sum_x
                with np.errstate(divide='ignore', invalid='ignore'):
                    slope_block = np.where(den != 0, num / den, np.nan)

                # aplicar máscara: pendiente solo donde hay >=2 datos válidos
                slope_block[n_valid < 2] = np.nan
                slope[s : s + n] = slope_block

            new_cols[f"{col}_tendency"] = slope

        # concatenar todo en el DataFrame final
        X_out = X.assign(**new_cols)
        return X_out

class CleanZerosTransformer(BaseEstimator, TransformerMixin):

    """
    Detecta pares de variables cVARIABLE (cantidad) y mVARIABLE (monto).
    Cuando la cantidad es 0, pone el monto en None en lugar de 0.
    
    Esto evita que el modelo aprenda relaciones incorrectas donde hay montos
    en 0 que en realidad deberían ser valores nulos (porque no existe la operación).
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
    
    def transform(self, X):
        """
        Pone en None los montos (mVARIABLE) cuando la cantidad (cVARIABLE) es 0.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
        
        X_transformed = X.copy()
        
        # Para cada par detectado, limpiar los montos cuando cantidad = 0
        for c_col, m_col in self.variable_pairs_:
            if c_col in X_transformed.columns and m_col in X_transformed.columns:
                # Crear máscara donde la cantidad es 0
                zero_mask = X_transformed[c_col] == 0
                
                # Poner el monto en None donde la cantidad es 0
                X_transformed.loc[zero_mask, m_col] = np.nan
        
        return X_transformed

class IntraMonthTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, exclude_cols=["foto_mes", "numero_de_cliente", "target", "label", "weight"]):
        self.exclude_cols = exclude_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        # Ratios
        X_transformed["ratio_rentabilidad_activos"] = X_transformed["mrentabilidad_annual"] / (X_transformed["mactivos_margen"] + 1)
        X_transformed["ratio_rentabilidad_pasivos"] = X_transformed["mrentabilidad_annual"] / (X_transformed["mpasivos_margen"] + 1)
        X_transformed["ratio_liquidez_relativa"] = X_transformed["mcuentas_saldo"] / (X_transformed["mpasivos_margen"] + 1)
        X_transformed["ratio_endeudamiento_prestamos"] = (X_transformed["mprestamos_personales"] + X_transformed["mprestamos_prendarios"] + X_transformed["mprestamos_hipotecarios"]) / (X_transformed["mactivos_margen"] + 1)
        X_transformed["ratio_intensidad_credito"] = X_transformed["mtarjeta_visa_consumo"] + X_transformed["mtarjeta_master_consumo"] / (X_transformed["Visa_mlimitecompra"] + X_transformed["Master_mlimitecompra"] + 1)
    
        # Totals
        X_transformed["total_limite_credito"] = X_transformed["Visa_mlimitecompra"] + X_transformed["Master_mlimitecompra"]

        X_transformed["total_prestamos"] = X_transformed["mprestamos_personales"] + X_transformed["mprestamos_prendarios"] + X_transformed["mprestamos_hipotecarios"]

        X_transformed["total_consumo_tarjetas"] = X_transformed["mtarjeta_visa_consumo"] + X_transformed["mtarjeta_master_consumo"]
    
        
        return X_transformed

class RandomForestFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_cols=["numero_de_cliente", "label", "weight"], n_estimators=20, num_leaves=16, min_data_in_leaf=100, feature_fraction_bynode=0.2, training_months= []):
        self.exclude_cols = exclude_cols
        self.training_months = training_months
        self.n_estimators = n_estimators
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
        X = X.copy()
        X_train = X.loc[X["foto_mes"].isin(self.training_months)]
        y = X_train["label"]
        X_train = X_train.drop(columns=self.exclude_cols)
        self.columns_ = X_train.columns

        dtrain = lgb.Dataset(
        data=X_train.values,
        label=y,
        free_raw_data=False)
        self.model_ = lgb.train(params=self.lgb_params, train_set=dtrain)
        logger.info("RandomForestFeaturesTransformer entrenado")
        
        return self
    
    def transform(self, X):
        X = X.copy()
        extra_cols = set(X.columns) - set(self.columns_)
        extra_cols = X[list(extra_cols)]
        X = X[self.columns_]

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