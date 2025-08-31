# Define hyperparameters
do_data_parallel = True

per_device_batch_size = 1
dataloader_num_workers = 2
learning_rate = 1e-4
epochs = 1

# Path PyTorch should save model to
tv_model_path = '/home/ubuntu/ml-model-training-assignment/model_data'

# Path Imagenette is at
imagenette_train_path = '/home/ubuntu/imagenette2/train'
imagenette_test_path = '/home/ubuntu/imagenette2/val'

# -1 means full dataset
train_data_size = 4
test_data_size = 4

device = 'cuda'

# Flag used to simulate limited memory. Set to 1.0 if you wish to use 100% memory on each device
memory_limit = 1.0

# Only use the specified devices
visible_devices = [0,1,2,3]
