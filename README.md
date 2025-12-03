# READ ME
Instrucciones para replicar experimento:
- ejecutar el notebook prod_00 (R)
- ejecutar el notebook prod_02 (R)
- Ejecutar el notebook entregas (python)

Los dos primeros en R entrenan los modelos, para generarlos utilicé 256 Gb de RAM y hubo momentos en los que se utilizó el swap. Luego de entrenar el primer modelo del semillerio se deja de utilizar la memoria swap y la velocidad del entrenamiento aumenta considerablemente. Probablemente se conseguiría entrenar en los mismos tiempos con menos RAM.
