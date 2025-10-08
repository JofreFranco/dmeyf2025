def calcular_clase_ternaria(df):
    df_clientes = df[["foto_mes", "numero_de_cliente"]].copy()
    meses = df_clientes['foto_mes'].unique()
    df_clientes["clase_ternaria"] = "CONTINUA"
    for n, mes in enumerate(meses):
        if n<len(meses)-1:
            baja_1 = set(df_clientes[(df_clientes['foto_mes'] == mes)]["numero_de_cliente"]) - set(df_clientes[(df_clientes['foto_mes'] == meses[n+1])]["numero_de_cliente"])
            df_clientes.loc[(df_clientes['numero_de_cliente'].isin(baja_1)) & (df_clientes['foto_mes'] == mes), 'clase_ternaria'] = "BAJA+1"

        if n<len(meses)-2:
            baja_2 = (set(df_clientes[(df_clientes['foto_mes'] == meses[n+1])]["numero_de_cliente"]) & set(df_clientes[(df_clientes['foto_mes'] == mes)]["numero_de_cliente"])) - set(df_clientes[(df_clientes['foto_mes'] == meses[n+2])]["numero_de_cliente"])
            df_clientes.loc[(df_clientes['numero_de_cliente'].isin(baja_2)) & (df_clientes['foto_mes'] == mes), 'clase_ternaria'] = "BAJA+2"
    df["clase_ternaria"] = df_clientes["clase_ternaria"]
    return df


def calcular_clase_binaria(df):
    """
    Calcula la clase binaria donde solo "BAJA+2" es positivo.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame que debe contener la columna 'clase_ternaria'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con la nueva columna 'clase_binaria' agregada
    """
    df = df.copy()
    
    # Verificar que existe la columna clase_ternaria
    if 'clase_ternaria' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'clase_ternaria'")
    
    # Crear clase binaria: solo BAJA+2 es positivo (1), el resto es negativo (0)
    df['clase_binaria'] = df['clase_ternaria'].apply(
        lambda x: 1 if x == 'BAJA+2' else 0
    )
    
    return df