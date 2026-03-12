from DataPreparationFunctions.get_sequences_start_end_indices import get_sequences_start_end_indices
def ensure_full_beat_start(masks_segment):
    S1_start, S1_end = get_sequences_start_end_indices(masks_segment[:,0])
    S2_start, S2_end = get_sequences_start_end_indices(masks_segment[:,4])
    
    if S1_start.size != 0 and S2_start.size != 0:
        border_limits = [S1_start[0].item(),S2_end[-1].item()]
        for i in range(masks_segment.shape[1]-1):
            masks_segment[0:border_limits[0],i] = 0
            masks_segment[border_limits[1]:-1,i] = 0
        
        masks_segment[0:border_limits[0],-1] = 1
        masks_segment[border_limits[1]:-1,-1] = 1
        
    else:
        for i in range(masks_segment.shape[1]-1):
            masks_segment[:,i] = 0
        masks_segment[:,-1] = 1
    return masks_segment