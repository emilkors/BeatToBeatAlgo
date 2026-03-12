import torch

def load_model(model, filepath, optimizer=None, device=None):
    """
    Load model weights and optionally optimizer state from a checkpoint.

    Parameters:
        model (torch.nn.Module): Model instance to load weights into.
        filepath (str): Path to the saved checkpoint.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state (default None).
        device (str or torch.device): Device to map the model (default 'cpu').

    Returns:
        model (torch.nn.Module): Model with loaded weights.
        optimizer (torch.optim.Optimizer or None): Optimizer with loaded state if provided.
        epoch (int or None): Epoch number saved in the checkpoint.
        loss (float or None): Loss saved in the checkpoint.
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    saved_model_name = checkpoint.get('model_name', None)

    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Important: move optimizer tensors to the same device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    epoch = checkpoint.get('epoch', None)
    loss = checkpoint.get('loss', None)
    print(f"Model '{saved_model_name}' loaded epoch: {epoch} with loss: {loss}")
    return model, saved_model_name, optimizer, epoch, loss