"""
Diccionario de datos: definiciones de columnas y tipos de datos
"""

# Columnas categóricas bancarias
BANK_CAT_COLS = [
    "active_quarter", 
    "cliente_vip", 
    "internet",
    "cdescubierto_preacordado", 
    "ccaja_seguridad", 
    "tcallcenter",
    "thomebanking", 
    "tmobile_app"
]

# Columnas categóricas de MasterCard
MASTER_CAT_COLS = [
    "Master_delinquency", 
    "Master_status"
]

# Columnas categóricas de Visa
VISA_CAT_COLS = [
    "Visa_delinquency", 
    "Visa_status"
]

# Columnas a excluir del procesamiento de features
EXCLUDE_COLS = [
    'numero_de_cliente', 
    'foto_mes', 
    'clase_ternaria', 
    'clase_binaria',
    'target_binario', 
    'clase_peso'
]

# Todas las columnas categóricas
ALL_CAT_COLS = BANK_CAT_COLS + MASTER_CAT_COLS + VISA_CAT_COLS

# Definición de períodos de tiempo para train/test
TRAIN_MONTHS = [202102, 202103]
TEST_MONTH = 202104

# Configuración de pesos para clases
PESO_BAJA_2 = 1.00002
PESO_BAJA_1 = 1.00001
PESO_CONTINUA = 1.0

# Configuración de ganancia para evaluación
GANANCIA_ACIERTO = 780000
COSTO_ESTIMULO = 20000

RANDOM_SEEDS = [537919, 923347, 173629, 419351, 287887]
