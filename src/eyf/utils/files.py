"""
Utilidades para manejo de archivos
"""
import os
import glob
import pandas as pd
import numpy as np

def convertir_a_nativo(obj):
        if isinstance(obj, dict):
            return {k: convertir_a_nativo(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convertir_a_nativo(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

def get_debug_filename(filename, debug_mode=False):
    """
    Genera el nombre del archivo con sufijo _DEBUG si está en modo debug.
    
    Parameters:
    -----------
    filename : str
        Nombre base del archivo
    debug_mode : bool, default=False
        Si True, agrega sufijo _DEBUG
        
    Returns:
    --------
    str
        Nombre del archivo con o sin sufijo _DEBUG
    """
    if not debug_mode:
        return filename
    
    # Separar nombre y extensión
    if '.' in filename:
        name, ext = filename.rsplit('.', 1)
        return f"{name}_DEBUG.{ext}"
    else:
        return f"{filename}_DEBUG"


def prob_to_prediction(input_csv_path, threshold=0.025, output_csv_path=None):
    """
    Convierte probabilidades en predicciones binarias aplicando un threshold.
    
    Parameters:
    -----------
    input_csv_path : str
        Ruta al archivo CSV con probabilidades
    threshold : float, default=0.025
        Threshold para convertir probabilidad en predicción (>=threshold = 1, <threshold = 0)
    output_csv_path : str, optional
        Ruta del archivo de salida. Si None, se genera automáticamente agregando "_prediction"
        
    Returns:
    --------
    str
        Ruta del archivo generado
    """
    # Leer archivo de probabilidades
    df = pd.read_csv(input_csv_path)
    
    if 'probabilidad' not in df.columns:
        raise ValueError(f"El archivo {input_csv_path} debe contener una columna 'probabilidad'")
    
    # Crear predicciones binarias
    df['Predicted'] = (df['probabilidad'] >= threshold).astype(int)
    
    # Generar nombre del archivo de salida si no se especifica
    if output_csv_path is None:
        base_path = input_csv_path.rsplit('.', 1)[0]
        output_csv_path = f"{base_path}_prediction.csv"
    
    # Guardar archivo con predicciones
    df_output = df[['numero_de_cliente', 'Predicted']].copy()
    df_output.to_csv(output_csv_path, index=False)
    
    num_predictions_1 = df['Predicted'].sum()
    total_predictions = len(df)
    percentage_1 = (num_predictions_1 / total_predictions) * 100
    
    print(f"✅ Predicciones generadas: {output_csv_path}")
    print(f"   Threshold usado: {threshold}")
    print(f"   Predicciones = 1: {num_predictions_1:,} ({percentage_1:.2f}%)")
    print(f"   Predicciones = 0: {total_predictions - num_predictions_1:,} ({100-percentage_1:.2f}%)")
    
    return output_csv_path


def process_experiment_predictions(experiment_dir, experiment_name, threshold=0.025):
    """
    Procesa todos los archivos de probabilidades en un directorio de experimento.
    
    Parameters:
    -----------
    experiment_dir : str
        Directorio del experimento
    threshold : float, default=0.025
        Threshold para las predicciones
    Returns:
    --------
    list
        Lista de archivos de predicción generados
    """
    # Construir patrón de búsqueda
    pattern = f"*{experiment_name}*.csv"
    # Buscar archivos de probabilidades
    experiment_dir = glob.escape(experiment_dir)
    search_pattern = os.path.join(experiment_dir, pattern)
    prob_files = glob.glob(search_pattern)
    
    # Filtrar solo archivos que contengan probabilidades (no predicciones ya generadas)
    prob_files = [f for f in prob_files if 'prediction' not in f and not f.endswith('_trials.csv')]
    
    if not prob_files:
        print(f"❌ No se encontraron archivos de probabilidades en {experiment_dir}")
        return []
    
    print(f"🔄 Procesando {len(prob_files)} archivos con threshold {threshold}")
    
    prediction_files = []
    for prob_file in prob_files:
        try:
            pred_file = prob_to_prediction(prob_file, threshold)
            prediction_files.append(pred_file)
        except Exception as e:
            print(f"❌ Error procesando {prob_file}: {e}")
    
    print(f"✅ {len(prediction_files)} archivos de predicción generados")
    return prediction_files
