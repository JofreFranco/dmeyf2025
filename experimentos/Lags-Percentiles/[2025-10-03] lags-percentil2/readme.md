# Experimento Lags-Percentiles

## Descripción
Este experimento evalúa el impacto de utilizar features con **lag 1** (variables del mes anterior) combinados con **percentiles** (features basadas en distribuciones históricas) para predecir la clase binaria de clientes.

## Objetivo
Determinar si la combinación de información histórica (lags) e información de distribuciones estadísticas (percentiles) mejora el poder predictivo respecto a usar solo lags.

## Preprocesamiento
- **Lag Transform**: Se calculan los valores lag 1 para todas las variables numéricas disponibles
- **Percentile Transform**: Se calculan features basadas en percentiles (25, 50, 75, 90, 95) de las variables originales
  - Features binarias: indica si está por encima del percentil
  - Features de distancia: distancia al valor del percentil
- **Manejo de NaN**: Los valores faltantes se mantienen como NaN para que LightGBM los maneje nativamente
- **Target**: Se utiliza clase binaria donde positivos = "BAJA+2" únicamente

## División de Datos
- **Train**: Meses 202101, 202102, 202103, 202104
- **Test**: Mes 202105  
- **Eval**: Mes 202106

## Optimización Bayesiana (Optuna)

### Configuración Inicial
- **Trials**: 50 iteraciones
- **Cross-Validation**: 5 folds estratificado
- **Semilla**: Primera semilla del diccionario (42)
- **Métrica objetivo**: Ganancia máxima

### Espacio de Búsqueda
- `num_leaves`: [8, 100]
- `learning_rate`: [0.005, 0.3]
- `min_data_in_leaf`: [5, 10000]  
- `feature_fraction`: [0.1, 1.0]
- `bagging_fraction`: [0.1, 1.0]

### Parámetros Fijos
- `max_bin`: 31
- `boosting_type`: gbdt
- `early_stopping_rounds`: 75 (CV), 200 (final)
- `num_threads`: 10

## Percentiles Configurados
- **P25**: Percentil 25 (primer cuartil)
- **P50**: Percentil 50 (mediana)
- **P75**: Percentil 75 (tercer cuartil)
- **P90**: Percentil 90
- **P95**: Percentil 95

Para cada percentil se generan:
1. Feature binaria: `{variable}_above_p{percentil}`
2. Feature de distancia: `{variable}_dist_p{percentil}`

## Arquitectura del Código
- **Formato script**: Sin definir funciones locales, código secuencial directo
- **Funciones reutilizables**: Localizadas en `src/eyf/utils.py` para uso en otros experimentos
- **Optimización modular**: Usa función genérica `optimize_hyperparameters_with_optuna()`
- **Pipeline personalizado**: Aplica LagTransformer primero, luego PercentileTransformer

## Replicación
Tras encontrar los mejores hiperparámetros, se entrena el modelo con las 10 semillas aleatorias definidas en `data_dict.py` para obtener métricas robustas y crear un ensemble final.

## Archivos Generados
- `lags_percentiles.json`: Mejores hiperparámetros encontrados
- `results_lags_percentiles.json`: Métricas por semilla y promedios
- `lags_percentiles_{seed}.csv`: Predicciones individuales por semilla
- `lags_percentiles_ensemble.csv`: Predicciones promediadas (ensemble final)
