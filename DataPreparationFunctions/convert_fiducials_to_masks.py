import numpy as np
from itertools import combinations
def convert_fiducials_to_masks(signal,manual_fiducials):
    """
    Function for converting manual fiducials for masks corresponding to signal's
    indices. The manual fiducial must include indices for fiducials
    (Cs, Ds, Es, Fs, Gs, Ks, Ls, Bd, Cd, Dd, Ed) in the signal for each beat in
    which it can be identified. 

    Parameters
    ----------
    signal : 1d array
        The signal that should be masked.
    manual_fiducials : pandas DataFrame (n_beats,n_fiducials)
        The DataFrame containing the fiducials (Cs, Ds, Es, Fs, Gs, Ks, Ls, Bd, Cd, Dd, Ed).

    Returns
    -------
    masks : ndarray (n_samples,n_masks_to_use)
        The mask matrix. Each column represent one mask, and every row 
        represents a sample.

    """
    
    
    # List of names
    S1_fiducials = ["Cs", "Ds", "Fs", "Gs", "Ks", "Ls"]
    # Generate all combinations and format as "AtoB"
    S1_combinations = [f"{a}_to_{b}" for a, b in combinations(S1_fiducials, 2)]
    
    S2_fiducials = ["Cd","Dd","Ed"]
    S2_combinations = [f"{a}_to_{b}" for a, b in combinations(S2_fiducials, 2)]
    
    # all intervals for predictions:
    combinations_all = S1_combinations + S2_combinations + ["None"] 
    
    extracted_intervals = {}
    #for combination in combinations_all[5:-1]:
    for combination in combinations_all[0:-1]:
        part_1 = combination.split('_')[0]
        part_2 = combination.split('_')[2]
        
        extracted_intervals[f"{combination}"] = np.asarray(
            [manual_fiducials[part_1].values, manual_fiducials[part_2].values]).T  # Assign a new key and value in each iteration
    
    # initialize masks
    masks = np.zeros((len(signal),len(combinations_all)))
    
    # run through each beat and set mask variable to 1 within the defined mask
    # segment from above
    for n_beat in range(0,len(manual_fiducials)):
        for idx, key in enumerate(combinations_all[:-1]):
            value = extracted_intervals[key]
            value = value.astype(int)
            masks[value[n_beat,0]:value[n_beat,1]+1,idx] = 1
                    
            
    # obtain the None mask (1 where all others are 0).
    masks[np.where(np.sum(masks[:,:len(extracted_intervals)-1], axis=1) == 0)[0],idx+1] = 1
    
    return masks, combinations_all