#!/bin/bash

# Script para ejecutar run_experiment.py secuencialmente con diferentes configuraciones
# Autor: Script generado automáticamente
# Fecha: $(date)

set -e  # Salir si cualquier comando falla

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir con colores
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar que el archivo run_experiment.py existe
if [ ! -f "run_experiment.py" ]; then
    print_error "El archivo run_experiment.py no existe en el directorio actual"
    exit 1
fi

# Lista de archivos de configuración
configs=("config1.yaml" "config2.yaml" "config3.yaml" "config4.yaml")

# Contador de experimentos
total_experiments=${#configs[@]}
current_experiment=0

print_info "Iniciando ejecución de $total_experiments experimentos secuencialmente"
print_info "Configuraciones a ejecutar: ${configs[*]}"
echo ""

# Crear directorio de logs si no existe
mkdir -p logs

# Timestamp para el log general
timestamp=$(date +"%Y%m%d_%H%M%S")
main_log="logs/experiments_${timestamp}.log"

print_info "Log principal: $main_log"
echo ""

# Función para ejecutar un experimento
run_experiment() {
    local config_file=$1
    local experiment_num=$2
    
    print_info "=========================================="
    print_info "Ejecutando experimento $experiment_num/$total_experiments"
    print_info "Configuración: $config_file"
    print_info "=========================================="
    
    # Verificar que el archivo de configuración existe
    if [ ! -f "$config_file" ]; then
        print_error "El archivo de configuración $config_file no existe"
        return 1
    fi
    
    # Timestamp para este experimento
    local exp_timestamp=$(date +"%Y%m%d_%H%M%S")
    local exp_log="logs/experiment_${config_file%.yaml}_${exp_timestamp}.log"
    
    print_info "Iniciando experimento con $config_file..."
    print_info "Log del experimento: $exp_log"
    
    # Ejecutar el experimento y capturar tanto stdout como stderr
    if poetry run python run_experiment.py --config "$config_file" 2>&1 | tee "$exp_log"; then
        print_success "Experimento $config_file completado exitosamente"
        echo "✅ $config_file - $(date)" >> "$main_log"
        return 0
    else
        print_error "El experimento $config_file falló"
        echo "❌ $config_file - $(date)" >> "$main_log"
        return 1
    fi
}

# Ejecutar cada experimento secuencialmente
failed_experiments=()
successful_experiments=()

for config in "${configs[@]}"; do
    current_experiment=$((current_experiment + 1))
    
    if run_experiment "$config" "$current_experiment"; then
        successful_experiments+=("$config")
        print_success "Experimento $config completado"
    else
        failed_experiments+=("$config")
        print_error "Experimento $config falló"
        
        # Preguntar si continuar con los siguientes experimentos
        echo ""
        read -p "¿Desea continuar con los siguientes experimentos? (y/n): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "Ejecución interrumpida por el usuario"
            break
        fi
    fi
    
    echo ""
    print_info "Esperando 5 segundos antes del siguiente experimento..."
    sleep 5
    echo ""
done

# Resumen final
echo ""
print_info "=========================================="
print_info "RESUMEN DE EJECUCIÓN"
print_info "=========================================="
print_info "Total de experimentos: $total_experiments"
print_info "Completados exitosamente: ${#successful_experiments[@]}"
print_info "Fallidos: ${#failed_experiments[@]}"

if [ ${#successful_experiments[@]} -gt 0 ]; then
    print_success "Experimentos exitosos:"
    for exp in "${successful_experiments[@]}"; do
        echo "  ✅ $exp"
    done
fi

if [ ${#failed_experiments[@]} -gt 0 ]; then
    print_error "Experimentos fallidos:"
    for exp in "${failed_experiments[@]}"; do
        echo "  ❌ $exp"
    done
fi

echo ""
print_info "Log principal guardado en: $main_log"
print_info "Logs individuales guardados en: logs/"

# Código de salida
if [ ${#failed_experiments[@]} -eq 0 ]; then
    print_success "Todos los experimentos se completaron exitosamente"
    exit 0
else
    print_warning "Algunos experimentos fallaron. Revisar logs para más detalles."
    exit 1
fi
