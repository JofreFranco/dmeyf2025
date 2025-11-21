debug_mode = True
VERBOSE = False
gcp = False
user = "emicardosomartinez"
experiment_name = "zlgbm-histfeatures"
dataset_path = "data/competencia_01_target.csv"
training_months = [201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908,
       201909, 201910, 201911, 201912, 202001, 202002, 202003, 202004,
       202005, 202006, 202007, 202008, 202009, 202010, 202011, 202012,
       202101, 202102, 202103, 202104, 202106, 202107]
save_model = True
eval_month = 202104
test_month = 202108
seeds = [537919, 923347, 173629, 419351, 287887, 1244, 24341, 1241, 4512, 6554, 62325, 6525235, 14, 4521, 474574, 74543, 32462, 12455, 5124, 55678]

sampling_rate = 0.02

fieldnames = ["experiment_name", "seed", "training_time", "moving_average_rev"]
features_to_drop = ["cprestamos_prendarios", "mprestamos_prendarios", "cprestamos_personales", "mprestamos_personales"]
canaritos = 5
gradient_bound = 0.1
n_seeds = 10
min_data_in_leaf = 20
params = {
    "canaritos": canaritos,
    "gradient_bound": gradient_bound,
    "feature_fraction": 0.50,
    "is_unbalance": False,
    "min_data_in_leaf": min_data_in_leaf,
}

if debug_mode:
    n_seeds = 1
    params["min_data_in_leaf"] = 2000
    params["gradient_bound"] = 0.4
    experiment_name += "_DEBUG"
    

# Agregar target y calcular weight
weight = {"BAJA+1": 1, "BAJA+2": 1.00002, "CONTINUA": 1}

if gcp:
    
    bucket_path = f'/home/{user}/buckets/b1/'
    results_file = f"{bucket_path}results.csv"
    experiment_log_file = f"{bucket_path}{experiment_name}.log"
else:
    bucket_path = "data/"
    results_file = "results.csv"
    experiment_log_file = "experiment.log"