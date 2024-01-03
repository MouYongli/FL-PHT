import json
import trainer as nnUNetTrainer 
import torch
import os

if __name__ == '__main__':
    BATCH_SIZE_TRAIN = 1
    BATCH_SIZE_VAL = 1
    NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', '2'))
    LR = 1e-2
    WEIGHT_DECAY = 3e-5
    PATCH_SIZE = [128, 128, 128]
    TRAINING_DATA_PATH = os.environ.get('TRAINING_DATA_PATH', '/home/data/MICCAI_FeTS2022_TrainingData')
    VALIDATION_DATA_PATH = os.environ.get('VALIDATION_DATA_PATH', '/home/data/MICCAI_FeTS2022_ValidationData')
    TRAINING_PARTITION_FILE = os.environ.get('TRAINING_PARTITION_FILE', 'partitioning_1.csv')
    VALIDATION_PARTITION_FILE = os.environ.get('VALIDATION_PARTITION_FILE', 'partitioning_1.csv')
    CLIENT_ID = int(os.environ.get('CLIENT_ID', '1'))
    DEVICE  = os.environ.get('DEVICE', 'cpu')

    parameters = {
        'batch_size_train': BATCH_SIZE_TRAIN,
        'batch_size_val': BATCH_SIZE_VAL,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LR,
        'weight_decay': WEIGHT_DECAY,
        'patch_size': PATCH_SIZE,
        'client_id': CLIENT_ID,
        'device': DEVICE
    }
    print(f'TRAINING_DATA_PATH: {TRAINING_DATA_PATH}')
    print(f'VALIDATION_DATA_PATH: {VALIDATION_DATA_PATH}')
    print(f'TRAINING_PARTITION_FILE: {TRAINING_PARTITION_FILE}')
    print(f'VALIDATION_PARTITION_FILE: {VALIDATION_PARTITION_FILE}')
    path = str(os.environ.get('FEDERATED_MODEL_PATH'))
    parameter_file = 'parameters.json'
    parameter_path = path + '/' + parameter_file
    # Save parameters to file
    with open(parameter_path, 'w') as f:
        json.dump(parameters, f)

    trainer = nnUNetTrainer.Trainer(client_id=CLIENT_ID, 
                                    training_batch_size=BATCH_SIZE_TRAIN, 
                                    validation_batch_size=BATCH_SIZE_VAL, 
                                    epochs=NUM_EPOCHS, 
                                    learning_rate=LR, 
                                    weight_decay=WEIGHT_DECAY, 
                                    patch_size=PATCH_SIZE,
                                    training_set_path=TRAINING_DATA_PATH,
                                    validation_set_path=VALIDATION_DATA_PATH,
                                    training_set_partition_file=TRAINING_PARTITION_FILE,
                                    validation_set_partition_file=VALIDATION_PARTITION_FILE,
                                    device=torch.device(DEVICE))
    trainer.run_training()
