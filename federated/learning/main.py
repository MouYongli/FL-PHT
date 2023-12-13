import mlflow
import trainer as nnUNetTrainer 
import torch
import os

base_dir = '/home/haneef/FL-PHT/FL-PHT/federated/learning/checkpoints'

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
    client_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Load the models
    for dir in client_dirs:
        model_path = os.path.join(base_dir, dir, 'model.pth')
        model = torch.load(model_path)
        models.append(model)
    return models

if __name__ == '__main__':
    BATCH_SIZE_TRAIN = 1
    BATCH_SIZE_VAL = 1
    NUM_EPOCHS = 2
    LR = 1e-2
    WEIGHT_DECAY = 3e-5
    PATCH_SIZE = [128, 128, 128]
    NUM_TRAINING_ROUNDS = 2

    mlflow.set_tracking_uri('https://mlflow.klee.informatik.rwth-aachen.de')
    mlflow.set_experiment('FeTS 2022')
    mlflow.start_run()
    mlflow.log_param('BATCH_SIZE_TRAIN', BATCH_SIZE_TRAIN)
    mlflow.log_param('BATCH_SIZE_VAL', BATCH_SIZE_VAL)
    mlflow.log_param('NUM_EPOCHS', NUM_EPOCHS)
    mlflow.log_param('LR', LR)
    mlflow.log_param('WEIGHT_DECAY', WEIGHT_DECAY)
    mlflow.log_param('PATCH_SIZE', PATCH_SIZE)
    mlflow.log_param('NUM_TRAINING_ROUNDS', NUM_TRAINING_ROUNDS)

    client_ids = [2, 9, 14, 19, 23]
    for i in range(NUM_TRAINING_ROUNDS):
        print(f'Training round {i+1} of {NUM_TRAINING_ROUNDS}')
        for client_id in client_ids:
            trainer = nnUNetTrainer.Trainer(client_id=client_id, training_batch_size=BATCH_SIZE_TRAIN, validation_batch_size=BATCH_SIZE_VAL, epochs=NUM_EPOCHS, learning_rate=LR, weight_decay=WEIGHT_DECAY, patch_size=PATCH_SIZE)
            trainer.run_training()
        
        # After each round, get the local models and compute the federated average
        models = get_local_models()
    
        # Get the federated average
        federated_avg = federated_average(models)
        torch.save(federated_avg, os.path.join(base_dir, 'model.pth'))
    
    mlflow.end_run()