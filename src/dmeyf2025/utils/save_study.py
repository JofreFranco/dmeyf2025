import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def save_trials(study, working_dir):
    trials_data = []
    for trial in study.trials:
        trial_dict = {
            'trial_number': trial.number,
            'value': trial.value,
            'state': trial.state.name,
            'best_iter': trial.user_attrs.get('best_iter', None),
            'duration_minutes': trial.user_attrs.get('duration_minutes', None),
            'auc': trial.user_attrs.get('AUC', None),
            'auc_std': trial.user_attrs.get('AUC-std', None),
            'gain': trial.user_attrs.get('Gain', None),
            'gain_std': trial.user_attrs.get('Gain-std', None),
            #'logloss': trial.user_attrs.get('Logloss', None),
            #'logloss_std': trial.user_attrs.get('Logloss-std', None),
        }
        # Agregar hiperparámetros
        trial_dict.update(trial.params)
        trials_data.append(trial_dict)
    
    trials_df = pd.DataFrame(trials_data)
    trials_csv_path = os.path.join(working_dir, f"trials.csv")
    trials_df.to_csv(trials_csv_path, index=False)
    logger.info(f"💾 Trials guardados en: {trials_csv_path}")