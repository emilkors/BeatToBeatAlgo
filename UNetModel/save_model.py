import torch
def save_model(model, filepath, model_name, optimizer=None, epoch=None, loss=None):
    """
    Save the model weights and optionally the optimizer state and training info.

    Parameters:
        model (torch.nn.Module): The model to save.
        filepath (str): Path to save the model.
        optimizer (torch.optim.Optimizer, optional): Optimizer to save state (default None).
        epoch (int, optional): Current epoch number (default None).
        loss (float, optional): Current loss value (default None).
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_name' : model_name
    }

    # Optionally save optimizer state
    if optimizer:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()

    # Optionally save epoch and loss information
    if epoch is not None:
        save_dict['epoch'] = epoch

    if loss is not None:
        save_dict['loss'] = loss

    # Save the state to the specified file
    torch.save(save_dict, filepath)
    print(f"Model '{model_name}' saved to {filepath}")