def scale(X, strategy="mean"):
    cols_to_scale = ["mrentabilidad_annual", "mcaja_ahorro", "mcaja_ahorro_adicional", "mcuentas_saldo", "mautoservicio", "mtarjeta_visa_consumo", "mtarjeta_master_consumo", "mpayroll", "mpayroll2", "mtransferencias_recibidas", "mtransferencias_emitidas", "mextraccion_autoservicio", "matm", "Master_msaldototal", "Master_msaldopesos", "Master_msaldodolares", "Visa_msaldototal", "Visa_msaldopesos", "Visa_msaldodolares", "Master_mconsumospesos", "Master_mconsumosdolares", "Visa_mconsumospesos", "Visa_mconsumosdolares", "Master_mconsumototal", "Visa_mconsumototal", "Master_mpagominimo", "Visa_mpagominimo"]
    if strategy == "mean":
        mean_by_mes = X[cols_to_scale + ["foto_mes"]][X["foto_mes"] != 202105].groupby("foto_mes").mean()
    elif strategy == "median":
        mean_by_mes = X[cols_to_scale + ["foto_mes"]][X["foto_mes"] != 202105].groupby("foto_mes").median()
    else:
        raise ValueError(f"Strategy {strategy} not supported")
    factors = mean_by_mes.pct_change(fill_method=None).loc[202106]
    factors = factors.fillna(0)
    for col in factors.index:
        X.loc[X["foto_mes"] == 202106, col] = X.loc[X["foto_mes"] == 202106, col] / (factors[col] + 1)
    return X