import logging
from os.path import exists
import pandas as pd
import os
import gc
import numpy as np
from sklearn.pipeline import Pipeline
from dmeyf2025.processors.sampler import SamplerProcessor
logger = logging.getLogger(__name__)


class ETL:
    """
    Clase ETL para procesar archivos CSV con transformers de sklearn.
    
    Esta clase permite leer un archivo CSV, aplicar una serie de procesadores
    de sklearn y guardar el resultado procesado.
    """

    def __init__(self, csv_directory: str, pipeline: Pipeline, blacklist_features: list = [], hard_filter: int = 0):
        """
        Inicializa la clase ETL.
        
        Args:
            csv_directory (str): Ruta al directorio del archivo CSV
            pipeline (Pipeline): Pipeline de sklearn para aplicar a los datos
            blacklist_features (list): Lista de features a eliminar
            hard_filter (int): Mes mínimo para considerar datos (antes de este mes no se usa ni para calcular históricos, etc.)
        """
        if exists(str(csv_directory).replace("crudo", "target")):
            self.csv_directory = str(csv_directory).replace("crudo", "target")
        else:
            self.csv_directory = csv_directory
            
        self.pipeline = pipeline
        self.data = None
        self.processed_data = None
        self.DEBUG = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
        self.blacklist_features = blacklist_features
        self.hard_filter = hard_filter
    def read_file(self) -> pd.DataFrame:
        """
        Lee el archivo CSV desde el directorio especificado.
        
        Returns:
            pd.DataFrame: DataFrame con los datos leídos del CSV
            
        Raises:
            FileNotFoundError: Si el archivo no existe
        """
        if self.DEBUG:
            csv_debug_directory = str(self.csv_directory).replace(".csv", "_debug.csv")
            if exists(csv_debug_directory):
                self.data = pd.read_csv(csv_debug_directory)
                logger.info(f"Archivo leído exitosamente: {len(self.data)} filas, {len(self.data.columns)} columnas")
                return self.data
        
        if not os.path.exists(self.csv_directory):
            raise FileNotFoundError(f"El archivo {self.csv_directory} no existe")
        
        try:
            self.data = pd.read_csv(self.csv_directory).drop(columns=self.blacklist_features)
            logger.info(f"Archivo leído exitosamente: {len(self.data)} filas, {len(self.data.columns)} columnas, se eliminaron {len(self.blacklist_features)} columnas")
            self.data = self.data[self.data['foto_mes'] >= self.hard_filter]
            logger.info(f"Se filtraron {len(self.data)} filas, {len(self.data.columns)} columnas")
            return self.data

        except Exception as e:
            raise Exception(f"Error al leer el archivo: {str(e)}")

    def process_data(self) -> tuple:
        """
        Procesa los datos aplicando el pipeline de sklearn.
        
        Returns:
            tuple: (X, y) con todos los datos procesados
            
        Raises:
            ValueError: Si no hay datos cargados
        """
        if self.data is None:
            raise ValueError("No hay datos cargados. Ejecuta leer_archivo() primero.")
        
        self.processed_data = self.pipeline.transform(self.data)
        
        logger.info(f"Procesamiento completado: {len(self.processed_data)} filas, {len(self.processed_data.columns)} columnas")
        
        # Separar X e y
        X = self.processed_data.drop(columns=["clase_ternaria"])
        y = self.processed_data["clase_ternaria"]
        
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Retorna los datos procesados.
        """
        return self.processed_data
    
    def save_processed_file(self, output_path: str, index: bool = False) -> None:
        """
        Guarda el archivo procesado en la ruta especificada.
        
        Args:
            output_path (str): Ruta donde guardar el archivo procesado
            index (bool): Si incluir el índice en el archivo guardado (default: False)
            
        Raises:
            ValueError: Si no hay datos procesados
        """
        if self.processed_data is None:
            raise ValueError("No hay datos procesados. Ejecuta procesar_datos() primero.")
        
        try:
            # Crear el directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Guardar el archivo
            self.processed_data.to_csv(output_path, index=index)
            logger.info(f"Archivo procesado guardado exitosamente en: {output_path}")
            
        except Exception as e:
            raise Exception(f"Error al guardar el archivo: {str(e)}")
    def split_data(self) -> pd.DataFrame:
        """
        Divide los datos en conjuntos de entrenamiento y prueba y kaggle.
        Args:
            train_months (list): Lista de meses de entrenamiento
            test_month (int): Mes de prueba
            eval_month (int): Mes de evaluación
        Returns:
            pd.DataFrame: DataFrame con los datos procesados
        """
        if self.train_months is not None:
            train_data = self.processed_data[self.processed_data['foto_mes'].isin(self.train_months)].copy()
            X_train, y_train = train_data.drop(columns=["clase_ternaria"]), train_data["clase_ternaria"]
        else:
            X_train, y_train = None, None
        if self.test_month is not None:
            test_data = self.processed_data[self.processed_data['foto_mes'] == self.test_month].copy()
            X_test, y_test = test_data.drop(columns=["clase_ternaria"]), test_data["clase_ternaria"]
        else:
            X_test, y_test = None, None
        if self.eval_month is not None:
            eval_data = self.processed_data[self.processed_data['foto_mes'] == self.eval_month].copy()
            X_eval, y_eval = eval_data.drop(columns=["clase_ternaria"]), eval_data["clase_ternaria"]
        else:
            X_eval, y_eval = None, None

        return X_train, y_train, X_test, y_test, X_eval, y_eval
    
    def execute_complete_pipeline(self, output_path: str = None, index: bool = False) -> tuple:
        """
        Ejecuta el pipeline completo: leer, procesar y guardar.
        
        Args:
            output_path (str): Ruta donde guardar el archivo procesado
            index (bool): Si incluir el índice en el archivo guardado (default: False)
            
        Returns:
            tuple: (X, y) con todos los datos procesados
        """
        logger.info("Iniciando pipeline ETL completo...")
        
        # Leer archivo
        self.read_file()
        
        # Procesar datos
        X, y = self.process_data()
        
        if output_path is not None:
            self.save_processed_file(output_path, index)
        
        logger.info("Pipeline ETL completado exitosamente!")
        return X, y

def prepare_data(df, training_months, eval_month, test_month, get_features, weight, sampling_rate):
    df["label"] = ((df["clase_ternaria"] == "BAJA+2") | (df["clase_ternaria"] == "BAJA+1")).astype(int)
    df["weight"] = np.array([weight[item] for item in df["clase_ternaria"]])
    df = df.drop(columns=["clase_ternaria"])
    df_transformed = get_features(df, training_months)
    del df
    gc.collect()
    if training_months is not None:
        df_train = df_transformed[df_transformed["foto_mes"].isin(training_months)]
    else:
        df_train = df_transformed[~df_transformed["foto_mes"].isin([eval_month, test_month])]
    df_eval = df_transformed[df_transformed["foto_mes"] == eval_month]
    df_test = df_transformed[df_transformed["foto_mes"] == test_month]
    del df_transformed
    gc.collect()
    y_eval, w_eval, X_eval = df_eval["label"], df_eval["weight"], df_eval.drop(columns=["label", "weight"])
    del df_eval
    gc.collect()
    y_test, w_test, X_test = df_test["label"], df_test["weight"], df_test.drop(columns=["label", "weight"])
    del df_test
    gc.collect()

    X_train, y_train = SamplerProcessor(sampling_rate).fit_transform(df_train.drop(columns=["label", "weight"]), df_train["label"])
    del df_train
    gc.collect()
    w_train = X_train["weight"]
    X_train = X_train.drop(columns=["weight"])
    return X_train, y_train, w_train, X_eval, y_eval, w_eval, X_test, y_test
if __name__ == "__main__":
    pass