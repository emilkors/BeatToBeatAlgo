from UNetModel.UNet import UNet
from UNetModel.load_model import load_model
FILEPATH = 'C:/Users/ZD94OW/OneDrive - Aalborg Universitet/SeismicHeart/Beat to Beat Fiducials/Trained Models/test_run'
model = UNet(1, 19)
model = load_model(model, FILEPATH, optimizer=None, device=None)