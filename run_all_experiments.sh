#!/bin/bash

# Script para ejecutar todos los experimentos .py en la carpeta experimentos y subcarpetas, en orden
# Uso: ./run_all_experiments.sh <config.yaml> [--debug | --no-debug]

# Función para mostrar ayuda
show_help() {
    echo "🚀 Script para ejecutar todos los experimentos"
    echo ""
    echo "Uso:"
    echo "  $0 <config.yaml> [--debug | --no-debug]"
    echo ""
    echo "Parámetros:"
    echo "  config.yaml    Archivo de configuración YAML (obligatorio)"
    echo "  --debug        Forzar modo debug para todos los experimentos"
    echo "  --no-debug     Forzar modo no-debug para todos los experimentos"
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
            echo "🐛 Modo debug activado para todos los experimentos"
            ;;
        --no-debug)
            DEBUG_FLAG="--no-debug"
            echo "🏃 Modo no-debug forzado para todos los experimentos"
            ;;
        *)
            echo "❌ Error: Argumento inválido '$2'"
            echo "Usa --debug o --no-debug, o ninguno"
            exit 1
            ;;
    esac
fi

echo "🚀 Iniciando ejecución de todos los experimentos..."
echo "📂 Buscando experimentos en 'eyf/experimentos'..."
echo "📄 Usando configuración: $YAML_CONFIG"

# Verificar que Poetry esté instalado
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry no está instalado. Instálalo primero: pip install poetry"
    exit 1
fi

# Cambiar al directorio del proyecto (donde está pyproject.toml)
cd "$(dirname "$0")"

# Verificar que estamos en el directorio correcto
if [ ! -f "pyproject.toml" ]; then
    echo "❌ No se encontró pyproject.toml. Asegúrate de ejecutar desde el directorio raíz del proyecto."
    exit 1
fi

echo "✅ Usando entorno virtual de Poetry"

# Primero, mostrar qué experimentos se van a ejecutar
echo ""
echo "🔍 Experimentos encontrados:"

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
echo "📊 Total: ${#experimentos[@]} experimentos"
echo "⚠️  ADVERTENCIA: Cada experimento puede tardar horas si DEBUG=False"
echo ""
read -p "¿Continuar? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Ejecución cancelada por el usuario."
    exit 0
fi

echo "=============================================="

# Ejecutar experimentos secuencialmente con información adicional
for i in "${!experimentos[@]}"; do
    archivo="${experimentos[$i]}"
    echo ""
    echo "🔄 Ejecutando ($((i+1))/${#experimentos[@]}): $archivo"
    echo "📅 Iniciado: $(date)"
    echo "----------------------------------------------"
    
    # Usar poetry run para ejecutar con el entorno virtual
    # Pasar el archivo YAML y los flags de debug si están configurados
    if [ -n "$DEBUG_FLAG" ]; then
        echo "🔧 Ejecutando: poetry run python \"$archivo\" \"$YAML_CONFIG\" $DEBUG_FLAG"
        poetry run python "$archivo" "$YAML_CONFIG" $DEBUG_FLAG
    else
        echo "🔧 Ejecutando: poetry run python \"$archivo\" \"$YAML_CONFIG\""
        poetry run python "$archivo" "$YAML_CONFIG"
    fi
    status=$?
    
    if [ $status -ne 0 ]; then
        echo "❌ Error al ejecutar $archivo (código de salida $status)"
        echo "💡 Tip: Revisa que DEBUG=True para experimentos de prueba"
        exit $status
    else
        echo "✅ Finalizado: $archivo"
        echo "📅 Completado: $(date)"
        
        # Pausa entre experimentos para liberar memoria
        if [ $((i+1)) -lt ${#experimentos[@]} ]; then
            echo "⏳ Pausa de 5 segundos para liberar memoria..."
            sleep 5
        fi
    fi
    echo "=============================================="
done

echo ""
echo "🎉 Todos los experimentos han sido ejecutados correctamente."
