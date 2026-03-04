#castIndex,wod_unique_cast,date,GMT_time,lat,lon,WMO_ID,z,Temperature,Salinity,Oxygen,Pressure,Chlorophyll,Nitrate,pH
from models.architectures.autoencoder import AE
from torch import nn, optim
import torch
model= AE()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_autoencoder(lr, cast_array, epochs):
    for iter in range(0, epochs):
        