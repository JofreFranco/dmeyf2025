import logging

def check_features(X_train, warning_treshold):
    """
    Logs a warning if any column in X_train has more than warning_treshold percentage of NaNs, Nones, or zeros.
    """
    logger = logging.getLogger(__name__)
    n_rows = len(X_train)
    for col in X_train.columns:
        col_data = X_train[col]
        nan_count = col_data.isna().sum()
        none_count = (col_data == None).sum() if col_data.dtype == object else 0  # Redundant if pandas treats None as np.nan, keeps for safety
        zero_count = (col_data == 0).sum()
        nan_none_total = nan_count + none_count

        nan_none_pct = nan_none_total / n_rows if n_rows > 0 else 0
        zeros_pct = zero_count / n_rows if n_rows > 0 else 0

        if nan_none_pct >= warning_treshold:
            logger.warning(f"Column '{col}' has {100*nan_none_pct:.2f}% NaN/None values ({nan_none_total} of {n_rows})")
        if zeros_pct >= warning_treshold:
            logger.warning(f"Column '{col}' has {100*zeros_pct:.2f}% zeros ({zero_count} of {n_rows})")
            # Check if all values in the column are the same (constant feature)
            if col_data.nunique(dropna=False) == 1:
                logger.warning(f"Column '{col}' is constant (all {n_rows} values are '{col_data.iloc[0]}').")
            
