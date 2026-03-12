import torch.nn as nn 
from torch import optim
import time
import os

def train_model(train_data, model,device, learning_rate, max_epochs):
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.BCELoss()
    
    total_start_time = time.time()
    
    model = model.to(device)
    
    for epoch in range(0,max_epochs):
        epoch_start_time = time.time()  # Record the start time of the current epoch
        model.train()
        running_loss = 0.0

        for batch_inputs, batch_masks in train_data:
            batch_inputs, batch_masks = batch_inputs.to(device), batch_masks.to(device)
            optimizer.zero_grad()
            output = model(batch_inputs)
            loss = criterion(output, batch_masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_epoch_loss = running_loss / len(train_data)  
        # Calculate the time taken for the current epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f'Epoch [{epoch+1}/{max_epochs}], Loss: {avg_epoch_loss:.10f}, '
              f'Epoch Time: {epoch_duration:.2f} seconds')

    # Record the total training time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"Total Training Time: {total_duration:.2f} seconds")
    return model, running_loss/len(train_data)

    
    
    