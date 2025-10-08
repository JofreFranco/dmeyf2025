#!/usr/bin/env python3
"""
Script para limpiar archivos de debug generados por los experimentos.
Elimina todos los archivos que contengan '_DEBUG' en su nombre.
"""

import os
import glob
from pathlib import Path

def cleanup_debug_files(base_path=None):
    """
    Elimina todos los archivos de debug del proyecto.
    
    Parameters:
    -----------
    base_path : str, optional
        Ruta base donde buscar archivos. Si None, usa el directorio del proyecto
    """
    if base_path is None:
        # Asumir que el script está en la raíz del proyecto
        base_path = Path(__file__).parent
    else:
        base_path = Path(base_path)
    
    print("🧹 INICIANDO LIMPIEZA DE ARCHIVOS DEBUG")
    print("=" * 50)
    
    # Patrones de archivos debug a eliminar
    debug_patterns = [
        "**/*_DEBUG.json",
        "**/*_DEBUG.csv", 
        "**/*_DEBUG*.csv",  # Para archivos como experiment_DEBUG_123.csv
        "**/*_DEBUG_trials.csv",
        "**/*_DEBUG_best_params.json"
    ]
    
    deleted_count = 0
    deleted_files = []
    
    for pattern in debug_patterns:
        files_to_delete = list(base_path.glob(pattern))
        
        for file_path in files_to_delete:
            try:
                if file_path.is_file():
                    print(f"🗑️  Eliminando: {file_path.relative_to(base_path)}")
                    file_path.unlink()
                    deleted_files.append(str(file_path.relative_to(base_path)))
                    deleted_count += 1
            except Exception as e:
                print(f"❌ Error eliminando {file_path}: {e}")
    
    print(f"\n✅ LIMPIEZA COMPLETADA!")
    print(f"📊 Archivos eliminados: {deleted_count}")
    
    if deleted_files:
        print(f"\n📁 Archivos eliminados:")
        for file in deleted_files:
            print(f"   - {file}")
    else:
        print("🎉 No se encontraron archivos debug para eliminar")
    
    return deleted_count, deleted_files


def cleanup_experiment_directory(experiment_dir):
    """
    Limpia archivos debug de un directorio específico de experimento.
    
    Parameters:
    -----------
    experiment_dir : str
        Ruta al directorio del experimento
    """
    experiment_path = Path(experiment_dir)
    
    if not experiment_path.exists():
        print(f"❌ Directorio no encontrado: {experiment_dir}")
        return 0, []
    
    print(f"🧹 Limpiando archivos debug en: {experiment_path}")
    
    debug_files = list(experiment_path.glob("*_DEBUG*"))
    deleted_count = 0
    deleted_files = []
    
    for file_path in debug_files:
        try:
            if file_path.is_file():
                print(f"🗑️  Eliminando: {file_path.name}")
                file_path.unlink()
                deleted_files.append(file_path.name)
                deleted_count += 1
        except Exception as e:
            print(f"❌ Error eliminando {file_path}: {e}")
    
    print(f"✅ Eliminados {deleted_count} archivos debug del directorio")
    return deleted_count, deleted_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Limpia archivos de debug generados por los experimentos"
    )
    parser.add_argument(
        "--dir",
        help="Directorio específico a limpiar (si no se especifica, limpia todo el proyecto)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo muestra qué archivos se eliminarían, sin eliminarlos"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 MODO DRY-RUN: Solo mostrando archivos que se eliminarían")
        print("Para eliminar realmente, ejecuta sin --dry-run")
        print("=" * 50)
        
        base_path = Path(args.dir) if args.dir else Path(__file__).parent
        debug_patterns = [
            "**/*_DEBUG.json",
            "**/*_DEBUG.csv", 
            "**/*_DEBUG*.csv",
            "**/*_DEBUG_trials.csv",
            "**/*_DEBUG_best_params.json"
        ]
        
        files_found = []
        for pattern in debug_patterns:
            files_found.extend(list(base_path.glob(pattern)))
        
        if files_found:
            print(f"📁 Se eliminarían {len(files_found)} archivos:")
            for file_path in files_found:
                print(f"   - {file_path.relative_to(base_path)}")
        else:
            print("🎉 No se encontraron archivos debug")
    
    elif args.dir:
        cleanup_experiment_directory(args.dir)
    else:
        cleanup_debug_files()
