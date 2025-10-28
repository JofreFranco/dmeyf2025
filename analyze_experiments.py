"""
Script para analizar resultados de experimentos usando test de Wilcoxon
"""
import logging
from dmeyf2025.utils.wilcoxon import analyze_experiments

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)

if __name__ == "__main__":

    analyze_experiments()

