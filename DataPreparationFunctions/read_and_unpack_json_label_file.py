import pandas as pd
import json
def read_and_unpack_json_label_file(filename):
    """
    Function reading and unpacking fiducial json file. 
    The file should follow the descriptions in fiducial_file_requirements.txt

    Parameters
    ----------
    filename : string
        The destination of the json file to read.

    Returns
    -------
    dfs : dictionary of DataFrames
        Dict with dataframes as elements. One for each type in the json file.
        Based on the keys in the .json file.
    data : dict
        The original read data from the data file in dict with lists for each
        channel and fiducial.
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    # Dictionary to store DataFrames
    dfs = {}

    for name, subdict in data.items():
        df = pd.DataFrame(subdict)
        dfs[name] = df  # Store the DataFrame with the subdictionary name as the key
    
    return dfs, data