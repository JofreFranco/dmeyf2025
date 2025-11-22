#!/bin/bash

# Script para ejecutar experimentos Python con poetry
# No se detiene si alguno falla y apaga el equipo al finalizar

# Registrar inicio
echo "=================================================="
echo "Iniciando ejecución de experimentos"
echo "Fecha y hora: $(date)"
echo "=================================================="

# Asegurarse de que el script continúe aunque haya errores
set +e

# Array con los archivos Python a ejecutar
archivos=(
    "zlgbm_historical2_ratio_lags.py"
    # Agregar más archivos aquí según sea necesario
)

# Contador de éxitos y fallos
exitosos=0
fallidos=0

# Ejecutar cada archivo
for archivo in "${archivos[@]}"; do
    echo ""
    echo "=================================================="
    echo "Ejecutando: $archivo"
    echo "Hora de inicio: $(date)"
    echo "=================================================="
    
    poetry run python "$archivo"
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ $archivo completado exitosamente"
        ((exitosos++))
    else
        echo "✗ $archivo falló con código de salida: $exit_code"
        ((fallidos++))
    fi
    
    echo "Hora de finalización: $(date)"
done

# Resumen final
echo ""
echo "=================================================="
echo "RESUMEN DE EJECUCIÓN"
echo "=================================================="
echo "Total de scripts: ${#archivos[@]}"
echo "Exitosos: $exitosos"
echo "Fallidos: $fallidos"
echo "Finalizado: $(date)"
echo "=================================================="

# Apagar el equipo
echo ""
echo "Apagando el equipo en 10 segundos..."
echo "Presiona Ctrl+C para cancelar"
sleep 10

# Apagar el sistema (requiere permisos de sudo)
sudo shutdown -h now

