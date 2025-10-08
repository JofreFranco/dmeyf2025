"""
Funciones para preprocessing y transformación de datos
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from ..utils.data_dict import ALL_CAT_COLS, EXCLUDE_COLS


def obtener_columnas_por_tipo(df):
    """
    Identifica las columnas numéricas y categóricas del DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a analizar
        
    Returns:
    --------
    dict
        Diccionario con las listas de columnas por tipo:
        - 'numeric': Lista de columnas numéricas
        - 'categorical': Lista de columnas categóricas
        - 'datetime': Lista de columnas de fecha/hora
        - 'object': Lista de columnas de tipo object (texto)
    """
    columnas_por_tipo = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'object': []
    }
    
    for col in df.columns:
        dtype = df[col].dtype
        
        # Identificar columnas categóricas conocidas
        if col in ALL_CAT_COLS:
            columnas_por_tipo['categorical'].append(col)
        # Columnas numéricas
        elif pd.api.types.is_numeric_dtype(dtype):
            columnas_por_tipo['numeric'].append(col)
        # Columnas de fecha/hora
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            columnas_por_tipo['datetime'].append(col)
        # Columnas de texto/object
        else:
            columnas_por_tipo['object'].append(col)
    
    # Imprimir resumen
    for tipo, columnas in columnas_por_tipo.items():
        if columnas:
            print(f"{tipo.title()}: {len(columnas)} columnas")
            if len(columnas) <= 10:
                print(f"  {columnas}")
            else:
                print(f"  {columnas[:5]} ... (+{len(columnas)-5} más)")
    
    return columnas_por_tipo


def transformar_columnas_categoricas(df, cat_cols, metodo='astype'):
    """
    Transforma las columnas categóricas usando el método especificado.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    cat_cols : list
        Lista de columnas categóricas a transformar
    metodo : str, default='astype'
        Método de transformación:
        - 'astype': Convertir a tipo category
        - 'onehot': One-hot encoding
        - 'label': Label encoding (ordinal)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con las columnas transformadas
    """
    df = df.copy()
    
    for col in cat_cols:
        if col not in df.columns:
            print(f"⚠️ Columna {col} no encontrada en el DataFrame")
            continue
            
        if metodo == 'astype':
            # Convertir a tipo category
            df[col] = df[col].astype('category')
            
        elif metodo == 'onehot':
            # One-hot encoding
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(df[[col]])
            
            # Crear nombres para las nuevas columnas
            feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
            
            # Eliminar columna original y agregar las nuevas
            df = df.drop(columns=[col])
            df = pd.concat([df, encoded_df], axis=1)
            
        elif metodo == 'label':
            # Label encoding (mapeo a números)
            unique_values = df[col].unique()
            label_map = {val: idx for idx, val in enumerate(unique_values)}
            df[col] = df[col].map(label_map)
                
    return df


def sample_dataset_estratificado(df, sample_ratio=1.0, debug_mode=False):
    """
    Realiza sampling estratificado del dataset, reduciendo solo la clase CONTINUA.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    sample_ratio : float, default=1.0
        Proporción de casos CONTINUA a mantener (0.0 a 1.0)
    debug_mode : bool, default=False
        Si True, mantiene solo 1000 casos CONTINUA independientemente de sample_ratio
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con sampling aplicado
    """
    if sample_ratio >= 1.0 and not debug_mode:
        return df.copy()
    
    df_sampled = df.copy()
    
    # Separar clases
    continua_mask = df_sampled['clase_ternaria'] == 'CONTINUA'
    other_classes = df_sampled[~continua_mask].copy()
    continua_cases = df_sampled[continua_mask].copy()
    
    # Determinar cuántos casos CONTINUA mantener
    if debug_mode:
        n_continua_keep = min(1000, len(continua_cases))
        print(f"🐛 DEBUG MODE: Manteniendo {n_continua_keep} casos CONTINUA")
    else:
        n_continua_keep = int(len(continua_cases) * sample_ratio)
        print(f"📊 SAMPLING: Manteniendo {n_continua_keep}/{len(continua_cases)} casos CONTINUA ({sample_ratio:.1%})")
    
    # Hacer sampling de casos CONTINUA
    if n_continua_keep < len(continua_cases):
        from sklearn.utils import resample
        continua_sampled = resample(
            continua_cases, 
            n_samples=n_continua_keep, 
            random_state=42,
            replace=False
        )
    else:
        continua_sampled = continua_cases
    
    # Combinar datasets
    df_final = pd.concat([other_classes, continua_sampled], ignore_index=True)
    
    print(f"✅ Dataset final: {len(df_final)} registros")
    print(f"   - BAJA+2: {(df_final['clase_ternaria'] == 'BAJA+2').sum()}")
    print(f"   - BAJA+1: {(df_final['clase_ternaria'] == 'BAJA+1').sum()}")  
    print(f"   - CONTINUA: {(df_final['clase_ternaria'] == 'CONTINUA').sum()}")
    
    return df_final


class LagTransformer(BaseEstimator, TransformerMixin):
    """
    Calcula lags de variables para n_lags meses anteriores.
    Si no hay información del mes anterior, se deja en nulo.
    """
    
    def __init__(self, n_lags=1, exclude_cols=None):
        """
        Parameters:
        -----------
        n_lags : int, default=1
            Número de lags a calcular
        exclude_cols : list, optional
            Columnas a excluir del cálculo de lags. Si None, usa EXCLUDE_COLS
        """
        self.n_lags = n_lags
        self.exclude_cols = exclude_cols if exclude_cols is not None else EXCLUDE_COLS
        self.lag_columns_ = None
        
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
    
    def __init__(self, n_deltas=1, exclude_cols=None):
        """
        Parameters:
        -----------
        n_deltas : int, default=1
            Número de deltas a calcular (1 = diferencia con mes anterior, 2 = con 2 meses atrás, etc.)
        exclude_cols : list, optional
            Columnas a excluir del cálculo de deltas. Si None, usa EXCLUDE_COLS
        """
        self.n_deltas = n_deltas
        self.exclude_cols = exclude_cols if exclude_cols is not None else EXCLUDE_COLS
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
    
    def __init__(self, variables=None, percentiles=[25, 50, 75, 90, 95], exclude_cols=None):
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
        self.exclude_cols = exclude_cols if exclude_cols is not None else EXCLUDE_COLS
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


class FullPreprocessor(BaseEstimator, TransformerMixin):
    """
    Procesador completo que combina lags, deltas y percentiles.
    """
    
    def __init__(self, n_lags=1, n_deltas=1, percentiles=[25, 50, 75, 90, 95], 
                 percentile_variables=None, exclude_cols=None):
        """
        Parameters:
        -----------
        n_lags : int, default=1
            Número de lags a calcular
        n_deltas : int, default=1
            Número de deltas a calcular
        percentiles : list, default=[25, 50, 75, 90, 95]
            Lista de percentiles a calcular
        percentile_variables : list, optional
            Variables para calcular percentiles
        exclude_cols : list, optional
            Columnas a excluir
        """
        self.n_lags = n_lags
        self.n_deltas = n_deltas
        self.percentiles = percentiles
        self.percentile_variables = percentile_variables
        self.exclude_cols = exclude_cols
        
        # Inicializar transformadores
        self.lag_transformer = LagTransformer(n_lags=n_lags, exclude_cols=exclude_cols)
        self.delta_transformer = DeltaTransformer(n_deltas=n_deltas, exclude_cols=exclude_cols)
        self.percentile_transformer = PercentileTransformer(
            variables=percentile_variables, 
            percentiles=percentiles, 
            exclude_cols=exclude_cols
        )
        
    def fit(self, X, y=None):
        """
        Entrena todos los transformadores.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
            
        # Ajustar transformadores en secuencia
        X_temp = X.copy()
        
        # 1. Ajustar y aplicar lags
        self.lag_transformer.fit(X_temp)
        X_temp = self.lag_transformer.transform(X_temp)
        
        # 2. Ajustar y aplicar deltas
        self.delta_transformer.fit(X_temp)
        X_temp = self.delta_transformer.transform(X_temp)
        
        # 3. Ajustar percentiles (solo en datos originales, no en lags/deltas)
        self.percentile_transformer.fit(X)
        
        return self
    
    def transform(self, X):
        """
        Aplica todas las transformaciones.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un pandas DataFrame")
            
        X_transformed = X.copy()
        
        # Aplicar transformaciones en secuencia
        X_transformed = self.lag_transformer.transform(X_transformed)
        X_transformed = self.delta_transformer.transform(X_transformed)
        X_transformed = self.percentile_transformer.transform(X_transformed)
        
        return X_transformed

