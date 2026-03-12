import numpy as np
def get_sequences_start_end_indices(mask):
    if mask.shape[0] == 1:
        mask=mask.reshape(-1)
        
    diff = np.diff(mask)
    start_indices = np.where(diff.T == 1)[0] + 1  # +1 to get the correct start index
    end_indices = np.where(diff == -1)[0]
    
    # Check if mask starts with a 1 (in case the first sequence starts from index 0)
    if mask[0] == 1:
        start_indices = np.insert(start_indices, 0, 0)
    
    # Check if mask ends with a 1 (in case the last sequence ends at the last element)
    if mask[-1] == 1:
        end_indices = np.append(end_indices, len(mask) - 1)
        
    return start_indices, end_indices