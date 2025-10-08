# Experimento Just-Lags

## Descripción
Este experimento evalúa el impacto de utilizar únicamente features con **lag 1** (variables del mes anterior) para predecir la clase binaria de clientes.

## Objetivo
Establecer una línea base del poder predictivo de las variables históricas con un solo período de retraso, sin agregar complejidad adicional.

## Preprocesamiento
- **Lag Transform**: Se calculan los valores lag 1 para todas las variables numéricas disponibles
- **Manejo de NaN**: Los valores faltantes se mantienen como NaN para que LightGBM los maneje nativamente
- **Target**: Se utiliza clase binaria donde positivos = "BAJA+1" y "BAJA+2"

## División de Datos
- **Train**: Meses 202101, 202102, 202103
- **Test**: Mes 202104  
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

## Arquitectura del Código
- **Formato script**: Sin definir funciones locales, código secuencial directo
- **Funciones reutilizables**: Localizadas en `src/eyf/utils.py` para uso en otros experimentos
- **Optimización modular**: Usa función genérica `optimize_hyperparameters_with_optuna()`

## Replicación
Tras encontrar los mejores hiperparámetros, se entrena el modelo con las 10 semillas aleatorias definidas en `data_dict.py` para obtener métricas robustas y crear un ensemble final.

## Archivos Generados
- `just_lags.json`: Mejores hiperparámetros encontrados
- `results_just_lags.json`: Métricas por semilla y promedios
- `just_lags_{seed}.csv`: Predicciones individuales por semilla
- `just_lags_ensemble.csv`: Predicciones promediadas (ensemble final)
