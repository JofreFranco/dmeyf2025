# Instalación:

Con poetry:
    ```
    poetry install
    ```
Python usado: 3.12.0

# Antes de correr:

En la carpeta data debe estar el archivo de los datos en formato csv, con el siguiente nombre: "competencia_01_crudo.csv"

# Para correr:

Para correr la búsqueda de hiperparámetros correr run_experiment.py, es el script principal con el cual experimenté. Este archivo guarda las predicciones, los mejores hiperparámetros y algunas cosas más dentro de la carpeta experiments. Los resultados de este script no son los que finalmente elegí en kaggle, sino el que sale de correr el notebook train_final.ipynb. La única diferencia entre estas predicciones y las que genera run_experiment.py es que en el notebook escalé algunas features con la intención de disminuir el data drifting de junio. train_final.ipynb puede correrse sin ejecutar la búsqueda de hiperparámetros porque dentro de la carpeta experiments ya están. Si se desea correr la búsqueda va a ser necesario modificar el nombre de la carpeta en donde están los resultados, de lo contrario, va a levantar un error.