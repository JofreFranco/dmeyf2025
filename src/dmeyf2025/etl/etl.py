import logging
from typing import List, Union

import pandas as pd
import os
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class ETL:
    """
    Clase ETL para procesar archivos CSV con transformers de sklearn.
    
    Esta clase permite leer un archivo CSV, aplicar una serie de procesadores
    de sklearn y guardar el resultado procesado.
    """
    
    def __init__(self, csv_directory: str, processors: List[Union[BaseEstimator, TransformerMixin]]):
        """
        Inicializa la clase ETL.
        
        Args:
            csv_directory (str): Ruta al directorio del archivo CSV
            processors (List[Union[BaseEstimator, TransformerMixin]]): 
                Lista de procesadores de sklearn para aplicar a los datos
        """
        self.csv_directory = csv_directory
        self.processors = processors
        self.data = None
        self.processed_data = None
        
    def read_file(self) -> pd.DataFrame:
        """
        Lee el archivo CSV desde el directorio especificado.
        
        Returns:
            pd.DataFrame: DataFrame con los datos leídos del CSV
            
        Raises:
            FileNotFoundError: Si el archivo no existe
        """
        if not os.path.exists(self.csv_directory):
            raise FileNotFoundError(f"El archivo {self.csv_directory} no existe")
        
        try:
            self.data = pd.read_csv(self.csv_directory)
            logger.info(f"Archivo leído exitosamente: {len(self.data)} filas, {len(self.data.columns)} columnas")
            return self.data

        except Exception as e:
            raise Exception(f"Error al leer el archivo: {str(e)}")
    
    def process_data(self) -> pd.DataFrame:
        """
        Procesa los datos aplicando la lista de procesadores de sklearn.
        
        Returns:
            pd.DataFrame: DataFrame con los datos procesados
            
        Raises:
            ValueError: Si no hay datos cargados
        """
        if self.data is None:
            raise ValueError("No hay datos cargados. Ejecuta leer_archivo() primero.")
        
        # Hacer una copia de los datos para no modificar el original
        self.processed_data = self.data.copy()
        
        # Aplicar cada procesador en secuencia
        for i, processor in enumerate(self.processors):
            try:
                logger.info(f"Aplicando procesador {i+1}/{len(self.processors)}: {type(processor).__name__}")
                
                # Si el procesador tiene fit, lo ajustamos primero
                if hasattr(processor, 'fit'):
                    processor.fit(self.processed_data)
                
                # Aplicar la transformación
                if hasattr(processor, 'transform'):
                    transformed_data = processor.transform(self.processed_data)
                    
                    # Si la transformación devuelve un array numpy, convertirlo a DataFrame
                    if hasattr(transformed_data, 'shape') and len(transformed_data.shape) == 2:
                        # Mantener los nombres de las columnas si es posible
                        if hasattr(processor, 'get_feature_names_out'):
                            column_names = processor.get_feature_names_out()
                        else:
                            column_names = [f'feature_{j}' for j in range(transformed_data.shape[1])]
                        
                        self.processed_data = pd.DataFrame(
                            transformed_data, 
                            columns=column_names,
                            index=self.processed_data.index
                        )
                    else:
                        # Si no es un array 2D, intentar mantener como DataFrame
                        self.processed_data = pd.DataFrame(transformed_data)
                        
            except Exception as e:
                raise Exception(f"Error al aplicar el procesador {type(processor).__name__}: {str(e)}")
        
        logger.info(f"Procesamiento completado: {len(self.processed_data)} filas, {len(self.processed_data.columns)} columnas")
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
        self.process_data()
        
        if output_path is not None:
            self.save_processed_file(output_path, index)
        
        logger.info("Pipeline ETL completado exitosamente!")
        return self.processed_data
if __name__ == "__main__":

    etl = ETL(csv_directory="./data/competencia_01_crudo.csv", processors=[])
    etl.ejecutar_pipeline_completo(output_path="./data/competencia_01_procesado.csv")