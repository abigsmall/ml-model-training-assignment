# Define hyperparameters
do_data_parallel = False

per_device_batch_size = [8, 16]
dataloader_num_workers = 2
learning_rate = 1e-4
epochs = 1

# Path PyTorch should save model to
tv_model_path = '/home/ubuntu/ml-model-training-assignment/model_data'

# Path Imagenette is at
imagenette_train_path = '/home/ubuntu/imagenette2/train'
imagenette_test_path = '/home/ubuntu/imagenette2/val'

# -1 means full dataset
train_data_size = 100
test_data_size = 20

device = 'cuda'

num_gpu = 1
num_samples = 1

# Flag used to simulate limited memory. Set to 1.0 if you wish to use 100% memory on each device
memory_limit = 1.0

# MLflow tracking
MLFLOW_TRACKING_URI = 'http://localhost:5001'
MLFLOW_EXPERIMENT_ID = '2'