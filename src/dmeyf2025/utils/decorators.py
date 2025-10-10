"""
Decoradores con argumentos de entrada.

Este módulo contiene ejemplos de decoradores que pueden recibir argumentos.
"""

from functools import wraps
import os
import pandas as pd

from typing import Callable


def save_data_decorator(filename: str):
    """
    Decorador que recibe filename, y si el archivo existe, lo lee, y si no, ejecuta la función y guarda el resultado en el archivo.
    
    Args:
        filename: Nombre del archivo a guardar
    
    Returns:
        datos procesados
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_filename = filename
            if os.path.exists(current_filename):
                return pd.read_csv(current_filename)
            else:
                result = func(*args, **kwargs)
                result.to_csv(current_filename, index=False)
                return result
        
        return wrapper
    return decorator

