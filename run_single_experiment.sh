#!/bin/bash

# Script para ejecutar un solo experimento específico
# Uso: ./run_single_experiment.sh <config.yaml> [--debug | --no-debug]

# Función para mostrar ayuda
show_help() {
    echo "🔍 Script para ejecutar un experimento específico"
    echo ""
    echo "Uso:"
    echo "  $0 <config.yaml> [--debug | --no-debug]"
    echo ""
    echo "Parámetros:"
    echo "  config.yaml    Archivo de configuración YAML (obligatorio)"
    echo "  --debug        Forzar modo debug para el experimento"
    echo "  --no-debug     Forzar modo no-debug para el experimento"
    echo ""
    echo "Ejemplos:"
    echo "  $0 experimento1.yaml"
    echo "  $0 experimento1.yaml --debug"
    echo "  $0 experimento1.yaml --no-debug"
    echo ""
}

# Verificar parámetros
if [ $# -eq 0 ]; then
    echo "❌ Error: Archivo de configuración YAML requerido"
    echo ""
    show_help
    exit 1
fi

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_help
    exit 0
fi

# Verificar que el archivo YAML existe
YAML_CONFIG="$1"
if [ ! -f "$YAML_CONFIG" ]; then
    echo "❌ Error: El archivo de configuración '$YAML_CONFIG' no existe"
    exit 1
fi

# Procesar argumentos adicionales
DEBUG_FLAG=""
if [ $# -gt 1 ]; then
    case "$2" in
        --debug)
            DEBUG_FLAG="--debug"
            echo "🐛 Modo debug activado para el experimento"
            ;;
        --no-debug)
            DEBUG_FLAG="--no-debug"
            echo "🏃 Modo no-debug forzado para el experimento"
            ;;
        *)
            echo "❌ Error: Argumento inválido '$2'"
            echo "Usa --debug o --no-debug, o ninguno"
            exit 1
            ;;
    esac
fi

echo "🔍 Experimentos disponibles:"
echo "📄 Usando configuración: $YAML_CONFIG"

# Método compatible con bash antiguo y zsh
experimentos=()
while IFS= read -r -d '' archivo; do
    experimentos+=("$archivo")
done < <(find experimentos -type f -name "*.py" -print0 | sort -z)

for i in "${!experimentos[@]}"; do
    # Extraer solo el nombre del archivo para display más limpio
    filename=$(basename "${experimentos[$i]}")
    dirname=$(dirname "${experimentos[$i]}")
    echo "  $((i+1)). $filename"
    echo "      📁 $dirname"
done

echo ""
read -p "Selecciona el número del experimento (1-${#experimentos[@]}): " selection

if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt "${#experimentos[@]}" ]; then
    echo "❌ Selección inválida."
    exit 1
fi

archivo="${experimentos[$((selection-1))]}"
echo ""
echo "🔄 Ejecutando: $archivo"
echo "📅 Iniciado: $(date)"
echo "=============================================="

# Verificar que Poetry esté disponible
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry no está instalado."
    exit 1
fi

# Ejecutar el experimento seleccionado con el archivo YAML y los flags
if [ -n "$DEBUG_FLAG" ]; then
    echo "🔧 Ejecutando: poetry run python \"$archivo\" \"$YAML_CONFIG\" $DEBUG_FLAG"
    poetry run python "$archivo" "$YAML_CONFIG" $DEBUG_FLAG
else
    echo "🔧 Ejecutando: poetry run python \"$archivo\" \"$YAML_CONFIG\""
    poetry run python "$archivo" "$YAML_CONFIG"
fi
status=$?

echo "=============================================="
if [ $status -ne 0 ]; then
    echo "❌ Error al ejecutar $archivo (código $status)"
    exit $status
else
    echo "✅ Finalizado exitosamente: $archivo"
    echo "📅 Completado: $(date)"
fi
