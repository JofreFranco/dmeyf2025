import logging
from os.path import exists
import pandas as pd
import os
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ETL:
    """
    Clase ETL para procesar archivos CSV con transformers de sklearn.
    
    Esta clase permite leer un archivo CSV, aplicar una serie de procesadores
    de sklearn y guardar el resultado procesado.
    """

    def __init__(self, csv_directory: str, pipeline: Pipeline, train_months: list = None, test_month: int = None, eval_month: int = None, blacklist_features: list = []):
        """
        Inicializa la clase ETL.
        
        Args:
            csv_directory (str): Ruta al directorio del archivo CSV
            pipeline (Pipeline): Pipeline de sklearn para aplicar a los datos
        """
        if exists(str(csv_directory).replace("crudo", "target")):
            self.csv_directory = str(csv_directory).replace("crudo", "target")
        else:
            self.csv_directory = csv_directory
            
        self.pipeline = pipeline
        self.data = None
        self.processed_data = None
        self.DEBUG = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
        self.train_months = train_months
        self.test_month = test_month
        self.eval_month = eval_month
        self.blacklist_features = blacklist_features
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
            return self.data

        except Exception as e:
            raise Exception(f"Error al leer el archivo: {str(e)}")

    def process_data(self) -> pd.DataFrame:
        """
        Procesa los datos aplicando el pipeline de sklearn.
        
        Returns:
            pd.DataFrame: DataFrame con los datos procesados
            
        Raises:
            ValueError: Si no hay datos cargados
        """
        if self.data is None:
            raise ValueError("No hay datos cargados. Ejecuta leer_archivo() primero.")
        
        self.processed_data = self.pipeline.transform(self.data)
        
        logger.info(f"Procesamiento completado: {len(self.processed_data)} filas, {len(self.processed_data.columns)} columnas")
        X_train, y_train, X_test, y_test, X_eval, y_eval = self.split_data()
        if X_train is not None:
            logger.info(f"DataFrame train: {len(X_train)}")
        if X_test is not None:
            logger.info(f"DataFrame test: {len(X_test)}")
        if X_eval is not None:
            logger.info(f"DataFrame eval: {len(X_eval)}")
        return X_train, y_train, X_test, y_test, X_eval, y_eval
    
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
    
    def execute_complete_pipeline(self, output_path: str = None, index: bool = False) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo: leer, procesar y guardar.
        
        Args:
            output_path (str): Ruta donde guardar el archivo procesado
            index (bool): Si incluir el índice en el archivo guardado (default: False)
            
        Returns:
            pd.DataFrame: DataFrame con los datos procesados
        """
        logger.info("Iniciando pipeline ETL completo...")
        
        # Leer archivo
        self.read_file()
        
        # Procesar datos
        X_train, y_train, X_test, y_test, X_eval, y_eval = self.process_data()
        
        if output_path is not None:
            self.save_processed_file(output_path, index)
        
        logger.info("Pipeline ETL completado exitosamente!")
        return X_train, y_train, X_test, y_test, X_eval, y_eval


if __name__ == "__main__":
    from sklearn.pipeline import Pipeline
    
    # Crear un pipeline vacío para el ejemplo
    empty_pipeline = Pipeline(steps=[])
    
    etl = ETL(csv_directory="./data/competencia_01_crudo.csv", pipeline=empty_pipeline)
    etl.execute_complete_pipeline(output_path="./data/competencia_01_procesado.csv")