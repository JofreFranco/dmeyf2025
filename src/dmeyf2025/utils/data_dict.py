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

# Configuración de ganancia para evaluación
GANANCIA_ACIERTO = 780000
COSTO_ESTIMULO = 20000

