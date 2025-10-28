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
    'clase_peso',
    'weight',
]

# Todas las columnas categóricas
ALL_CAT_COLS = BANK_CAT_COLS + MASTER_CAT_COLS + VISA_CAT_COLS

# Configuración de ganancia para evaluación
GANANCIA_ACIERTO = 780000
COSTO_ESTIMULO = 20000

FINANCIAL_COLS = [
"mrentabilidad",
"mrentabilidad_annual",
"mcomisiones",
"mactivos_margen",
"mpasivos_margen",
"mcuenta_corriente_adicional",
"mcuenta_corriente",
"mcaja_ahorro",
"mcaja_ahorro_adicional",
"mcaja_ahorro_dolares",
"mcuentas_saldo",
"mautoservicio",
"mtarjeta_visa_consumo",
"mtarjeta_master_consumo",
"mprestamos_personales",
"mprestamos_prendarios",
"mprestamos_hipotecarios",
"mplazo_fijo_dolares",
"mplazo_fijo_pesos",
"minversion1_pesos",
"minversion1_dolares",
"minversion2",
"mpayroll",
"mpayroll2",
"mcuenta_debitos_automaticos",
"mttarjeta_visa_debitos_automaticos",
"mttarjeta_master_debitos_automaticos",
"mpagodeservicios",
"mpagomiscuentas",
"mcajeros_propios_descuentos",
"mtarjeta_visa_descuentos",
"mtarjeta_master_descuentos",
"mcomisiones_mantenimiento",
"mcomisiones_otras",
"mforex_buy",
"mforex_sell",
"mtransferencias_recibidas",
"mtransferencias_emitidas",
"mextraccion_autoservicio",
"mcheques_depositados",
"mcheques_emitidos",
"mcheques_depositados_rechazados",
"mcheques_emitidos_rechazados",
"matm",
"matm_other",
"Master_mfinanciacion_limite",
"Master_msaldototal",
"Master_msaldopesos",
"Master_msaldodolares",
"Master_mconsumospesos",
"Master_mconsumosdolares",
"Master_mlimitecompra",
"Master_madelantopesos",
"Master_madelantodolares",
"Master_mpagado",
"Master_mpagospesos",
"Master_mpagosdolares",
"Master_mconsumototal",
"Master_mpagominimo",
"Visa_mfinanciacion_limite",
"Visa_msaldototal",
"Visa_msaldopesos",
"Visa_msaldodolares",
"Visa_mconsumospesos",
"Visa_mconsumosdolares",
"Visa_mlimitecompra",
"Visa_madelantopesos",
"Visa_madelantodolares",
"Visa_mpagado",
"Visa_mpagospesos",
"Visa_mpagosdolares",
"Visa_mconsumototal",
"Visa_mpagominimo",]
