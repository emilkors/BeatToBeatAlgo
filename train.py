import torch
from TrainFunctions.H5DataLoader import H5DataLoader
from TrainFunctions.train_model import train_model
from UNetModel.UNet import UNet
from UNetModel.save_model import save_model

TRAIN_FILE = "C:/Users/ZD94OW/SeismicHeartLocal/Code/BeatToBeatAlgo/train_data.h5"
MODEL_DESTINATION = "C:/Users/ZD94OW/OneDrive - Aalborg Universitet/SeismicHeart/Beat to Beat Fiducials/Trained Models/test_run"
MAX_EPOCHS = 11

# Data preparation
data_loader_train = H5DataLoader("Train",TRAIN_FILE)
data_loader_train.prepare_unet_tensor_dataset(batch_size=12, shuffle=True)

unet = UNet(1,19)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001


t_m, last_epoch_loss = train_model(data_loader_train.tensor_data_loader_unet, unet, device, learning_rate, MAX_EPOCHS)
save_model(t_m, MODEL_DESTINATION, "test_run", None, MAX_EPOCHS, last_epoch_loss)


