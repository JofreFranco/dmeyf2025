"""
Script para analizar resultados de experimentos usando test de Wilcoxon
"""
import argparse
import logging
from pathlib import Path
from dmeyf2025.utils.wilcoxon import analyze_experiments

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analizar resultados de experimentos con test de Wilcoxon"
    )
    parser.add_argument(
        '--tracking-file',
        type=str,
        default='results/experiments_tracking.csv',
        help='Path al archivo de tracking de experimentos'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Nivel de significancia para el test de Wilcoxon'
    )
    args = parser.parse_args()
    
    analyze_experiments(tracking_file=args.tracking_file, alpha=args.alpha)

