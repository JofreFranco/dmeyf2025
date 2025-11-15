#!/bin/bash

################################################################################
# Script para ejecutar experimentos de Python secuencialmente
# 
# Este script ejecuta una lista de scripts de Python en orden, continúa
# ante errores, y apaga la máquina al finalizar.
#
# Uso: ./run_experiments.sh
################################################################################

# ============================================================================
# CONFIGURACIÓN - Editar esta sección para cambiar los scripts a ejecutar
# ============================================================================

SCRIPTS=(
    "zlgbm_baseline.py"
    "zlgbm_monthly.py"
    "zlgbm_historical_features.py"
    #"zlgbm_historical_features_is_unbalanced.py"
)

# ============================================================================
# CONFIGURACIÓN ADICIONAL
# ============================================================================


WORK_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_FILE="${WORK_DIR}/experiments_run.log"
PYTHON_CMD="poetry run python"

# ============================================================================
# FUNCIONES
# ============================================================================

# Función para imprimir con timestamp
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] ${message}"
    
    # También escribir al archivo de log si está configurado
    if [ -n "$LOG_FILE" ]; then
        echo "[${timestamp}] ${message}" >> "$LOG_FILE"
    fi
}

# Función para imprimir separador
print_separator() {
    log_message "================================================================================"
}

# ============================================================================
# INICIO DEL SCRIPT
# ============================================================================

print_separator
log_message "INICIO DE EJECUCIÓN DE EXPERIMENTOS"
log_message "Directorio de trabajo: ${WORK_DIR}"
log_message "Total de scripts a ejecutar: ${#SCRIPTS[@]}"
print_separator

# Cambiar al directorio de trabajo
cd "$WORK_DIR" || {
    log_message "ERROR: No se pudo cambiar al directorio ${WORK_DIR}"
    exit 1
}

# Contadores
TOTAL_SCRIPTS=${#SCRIPTS[@]}
SUCCESSFUL=0
FAILED=0

# Timestamp de inicio
START_TIME=$(date +%s)

# ============================================================================
# EJECUCIÓN DE SCRIPTS
# ============================================================================

for i in "${!SCRIPTS[@]}"; do
    SCRIPT="${SCRIPTS[$i]}"
    SCRIPT_NUM=$((i + 1))
    
    print_separator
    log_message "Ejecutando script ${SCRIPT_NUM}/${TOTAL_SCRIPTS}: ${SCRIPT}"
    print_separator
    
    # Verificar que el script existe
    if [ ! -f "$SCRIPT" ]; then
        log_message "WARNING: El archivo ${SCRIPT} no existe. Saltando..."
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Timestamp de inicio del script
    SCRIPT_START=$(date +%s)
    
    # Ejecutar el script (continuar ante errores)
    if $PYTHON_CMD "$SCRIPT"; then
        SCRIPT_END=$(date +%s)
        SCRIPT_DURATION=$((SCRIPT_END - SCRIPT_START))
        SUCCESSFUL=$((SUCCESSFUL + 1))
        log_message "✓ Script ${SCRIPT} completado exitosamente en ${SCRIPT_DURATION} segundos"
    else
        EXIT_CODE=$?
        SCRIPT_END=$(date +%s)
        SCRIPT_DURATION=$((SCRIPT_END - SCRIPT_START))
        FAILED=$((FAILED + 1))
        log_message "✗ Script ${SCRIPT} falló con código de salida ${EXIT_CODE} después de ${SCRIPT_DURATION} segundos"
        log_message "  Continuando con el siguiente script..."
    fi
done

# ============================================================================
# RESUMEN FINAL
# ============================================================================

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

print_separator
log_message "RESUMEN DE EJECUCIÓN"
print_separator
log_message "Scripts ejecutados: ${TOTAL_SCRIPTS}"
log_message "Exitosos: ${SUCCESSFUL}"
log_message "Fallidos: ${FAILED}"
log_message "Tiempo total: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
print_separator

# ============================================================================
# APAGADO DEL SISTEMA
# ============================================================================

log_message "Todos los experimentos han finalizado."
log_message "Apagando el sistema en 10 segundos..."
log_message "Presiona Ctrl+C para cancelar el apagado."

# Esperar 10 segundos para dar tiempo a cancelar si es necesario
sleep 10

log_message "Ejecutando apagado del sistema..."
sudo shutdown -h now

