import torch
import os
import mlflow
import json
import numpy as np

def federated_average(models):
    """
    This function takes a list of models (state_dicts) and returns the federated average.

    :param models: list of state_dicts
    :return: federated average model
    """
    # Initialize a dictionary to hold the sum of the weights
    weight_sum = None

    for model in models:
        if weight_sum is None:
            weight_sum = model
        else:
            for key in weight_sum.keys():
                weight_sum[key] += model[key]

    # Divide the sum by the number of models to get the average
    for key in weight_sum.keys():
        weight_sum[key] = torch.div(weight_sum[key], len(models))

    return weight_sum

def get_local_models():
    models = []

    # Get the list of client directories
    aggregation_sources = str(os.environ['FEDERATED_AGGREGATION_PATHS'])

    # Load the models from each client directory
    for dir in aggregation_sources.split(","): 
        print(os.listdir(dir))
        model_path = dir + "/model.pth"
        print(f"Loading model from {model_path}")
        # for file_name in os.listdir(dir):
        model = torch.load(model_path, map_location=torch.device('cpu'))
        models.append(model)
    return models


def process_params(param_file_path):
    """
    This function takes a path to a parameter file, reads the parameters and logs them to mlflow.

    :param param_file_path: path to parameter file
    :return: None
    """
    with open(param_file_path, 'r') as f:
        params = json.load(f)
    client_id = params['client_id']

    for key, value in params.items():
        mlflow.log_param(f'{key}_{client_id}', value)

def process_losses(loss_file_path):
    """
    This function takes a path to a loss file, reads the losses and logs them to mlflow.

    :param loss_file_path: path to loss file
    :return: None
    """
    with open(loss_file_path, 'r') as f:
        losses = json.load(f)
    client_id = losses['client_id']

    for epoch, loss in enumerate(losses['train_losses']):
        mlflow.log_metric(f'epoch_loss_{client_id}', loss, epoch)
    
    mlflow.log_metric(f'train_loss_{client_id}', np.mean(losses['train_losses']))
    mlflow.log_metric(f'val_loss_{client_id}', losses['val_loss'])
    mlflow.log_metric(f'val_mean_fg_dice_{client_id}', losses['val_mean_fg_dice'])
    for i in range(len(losses['val_dice_per_class_or_region'])):
        mlflow.log_metric(f'val_dice_per_class_or_region_{i}_{client_id}', losses['val_dice_per_class_or_region'][i])

def process_accuracy(accuracy_file_path):
    """
    This function takes a path to an accuracy file, reads the accuracy and logs it to mlflow.

    :param accuracy_file_path: path to accuracy file
    :return: None
    """
    with open(accuracy_file_path, 'r') as f:
        accuracy = json.load(f)
    client_id = accuracy['client_id']

    mlflow.log_metric(f'accuracy_{client_id}', accuracy['accuracy'])
    
def get_run_id_if_exists():
    """
    This function returns the run_id if there exists a file called run_id.txt in the global federated directory.

    :return: run_id or None
    """
    path = str(os.environ['FEDERATED_GLOBAL_STORAGE'])
    run_id_file = path + "/run_id.txt"
    if os.path.isfile(run_id_file):
        with open(run_id_file, 'r') as f:
            run_id = f.read()
    else:
        run_id = None
    return run_id

def create_mlflow_run():
    """
    This function creates a new mlflow run and returns the run_id.

    :return: run_id
    """
    mlflow.set_tracking_uri("https://mlflow.klee.informatik.rwth-aachen.de")
    mlflow.set_experiment("FeTS 2022")
    parent_run_id = get_run_id_if_exists()
    if parent_run_id is None:
        mlflow.start_run()  
        parent_run_id = mlflow.active_run().info.run_id
        path = str(os.environ['FEDERATED_GLOBAL_STORAGE'])
        run_id_file = path + "/run_id.txt"
        with open(run_id_file, 'w') as f:
            f.write(parent_run_id)
    
    return parent_run_id

def log_everything_to_mlflow():
    """
    This function logs all files in the global federated directory to mlflow.

    :return: None
    """
    paths = str(os.environ['FEDERATED_AGGREGATION_PATHS'])
    for path in paths.split(","):
        for file_name in os.listdir(path):
            file_path = path + "/" + file_name
            if file_name == "parameters.json":
                process_params(file_path)
            elif file_name == "losses.json":
                process_losses(file_path)
            elif file_name == "accuracy.json":
                process_accuracy(file_path)

if __name__ == '__main__':
    # Create a new mlflow run
    parent_run_id = create_mlflow_run()

    with mlflow.start_run(run_id=parent_run_id, nested=True):
        log_everything_to_mlflow()

        models = get_local_models()
        federated_average_model = federated_average(models)

        model_path = str(os.environ['FEDERATED_MODEL_PATH'])
        torch.save(federated_average_model, f"{model_path}/model.pth")