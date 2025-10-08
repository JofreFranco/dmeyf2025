# Experimento Delta-Lags

## Descripción
Este experimento evalúa el impacto de utilizar features con **delta 1** (diferencias respecto al mes anterior) combinados con **lag 1** (variables del mes anterior) para predecir la clase binaria de clientes.

## Objetivo
Determinar si la combinación de información de cambios (deltas) e información histórica (lags) mejora el poder predictivo respecto a usar solo lags.

## Preprocesamiento
- **Delta Transform**: Se calculan las diferencias (delta 1) para todas las variables numéricas disponibles
- **Lag Transform**: Se calculan los valores lag 1 para todas las variables numéricas disponibles (aplicado después del delta)
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

## Arquitectura del Código
- **Formato script**: Sin definir funciones locales, código secuencial directo
- **Funciones reutilizables**: Localizadas en `src/eyf/utils.py` para uso en otros experimentos
- **Optimización modular**: Usa función genérica `optimize_hyperparameters_with_optuna()`
- **Pipeline personalizado**: Aplica DeltaTransformer primero, luego LagTransformer

## Replicación
Tras encontrar los mejores hiperparámetros, se entrena el modelo con las 10 semillas aleatorias definidas en `data_dict.py` para obtener métricas robustas y crear un ensemble final.

## Archivos Generados
- `delta_lags.json`: Mejores hiperparámetros encontrados
- `results_delta_lags.json`: Métricas por semilla y promedios
- `delta_lags_{seed}.csv`: Predicciones individuales por semilla
- `delta_lags_ensemble.csv`: Predicciones promediadas (ensemble final)
