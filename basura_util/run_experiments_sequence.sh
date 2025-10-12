#!/bin/bash

# Script para ejecutar experimentos secuencialmente
# Los experimentos se ejecutan en orden: run_experiment3.py, run_experiment2.py, run_experiment.py

set -e  # Detener si hay algún error

echo "========================================"
echo "Iniciando secuencia de experimentos"
echo "========================================"
echo ""

# Experimento 3
echo "----------------------------------------"
echo "Ejecutando run_experiment3.py..."
echo "----------------------------------------"
poetry run python run_experiment3.py --config config7.yaml
if [ $? -eq 0 ]; then
    echo "✓ run_experiment3.py completado exitosamente"
else
    echo "✗ Error en run_experiment3.py"
    exit 1
fi
echo ""


# Experimento 2
echo "----------------------------------------"
echo "Ejecutando run_experiment2.py..."
echo "----------------------------------------"
poetry run python run_experiment2.py --config config8.yaml
if [ $? -eq 0 ]; then
    echo "✓ run_experiment2.py completado exitosamente"
else
    echo "✗ Error en run_experiment2.py"
    exit 1
fi
echo ""

# Experimento 1
echo "----------------------------------------"
echo "Ejecutando run_experiment.py..."
echo "----------------------------------------"
poetry run python run_experiment.py --config config6.yaml
if [ $? -eq 0 ]; then
    echo "✓ run_experiment.py completado exitosamente"
else
    echo "✗ Error en run_experiment.py"
    exit 1
fi
echo ""

echo "----------------------------------------"
echo "Ejecutando run_experiment3.py..."
echo "----------------------------------------"
poetry run python run_experiment3.py --config config4.yaml
if [ $? -eq 0 ]; then
    echo "✓ run_experiment3.py completado exitosamente"
else
    echo "✗ Error en run_experiment3.py"
    exit 1
fi
echo ""
# Experimento 2
echo "----------------------------------------"
echo "Ejecutando run_experiment2.py..."
echo "----------------------------------------"
poetry run python run_experiment2.py --config config9.yaml
if [ $? -eq 0 ]; then
    echo "✓ run_experiment2.py completado exitosamente"
else
    echo "✗ Error en run_experiment2.py"
    exit 1
fi
echo ""
echo "----------------------------------------"
echo "Ejecutando run_experiment.py..."
echo "----------------------------------------"
poetry run python run_experiment.py --config config5.yaml
if [ $? -eq 0 ]; then
    echo "✓ run_experiment.py completado exitosamente"
else
    echo "✗ Error en run_experiment.py"
    exit 1
fi
echo ""

echo "========================================"
echo "Todos los experimentos completados ✓"
echo "========================================"

