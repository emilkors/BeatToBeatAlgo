import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import h5py

from LoadingModule import Recording
from LoadingModule.LoadingTools.ChannelSettingsReader import ChannelSettingsReader
from DataPreparationFunctions.read_and_unpack_json_label_file import read_and_unpack_json_label_file
from DataPreparationFunctions.downsample_data import downsample_data
from DataPreparationFunctions.get_message_from_json import get_message
from DataPreparationFunctions.convert_fiducials_to_masks import convert_fiducials_to_masks
from DataPreparationFunctions.ensure_full_beat_start import ensure_full_beat_start
from DataPreparationFunctions.create_h5_if_not_exists import create_h5_if_not_exists

ORIGINAL_SAMPLE_RATE = 5000
TARGET_SAMPLE_RATE = 1000
SEGMENT_LENGTH = 10
OVERLAP_LENGTH = 2
TARGET_H5 = 'C:/Users/ZD94OW/SeismicHeartLocal/Code/BeatToBeatAlgo/train_data.h5'
SCALER = MinMaxScaler(feature_range=(0, 1))

chldr_sh = ChannelSettingsReader("ChannelSettings.xml")
data_all = pd.read_excel('C:/Users/ZD94OW/SeismicHeartLocal/Code/BeatToBeatAlgo/data_to_train.xlsx')
targets = ["SCG sternum AC", "SCG sternum Z"]

create_h5_if_not_exists(TARGET_H5)

for index, row in data_all.iterrows():
    manual_fiducials = read_and_unpack_json_label_file(row["cursor_file_dest"])[0]
    idx = next(i for i, v in enumerate(list(manual_fiducials.keys())) if v in targets)
    ch_to_use = list(manual_fiducials.keys())[idx]
    manual_fiducials = manual_fiducials[ch_to_use]
    
    if not pd.isna(row["data_mat_file"]):
        rec = Recording.from_mat(row["data_mat_file"])
        rec.SetSignalHandlers(None,0,0,0)
        name = row["data_mat_file"].split('\\')[-1].split('.')[0]
    elif not pd.isna(row["data_ndjson"]):
        msg = get_message(row["data_index_json"], row["data_ndjson"], row["message_id"])
        body = msg["Body"]
        name = row["data_ndjson"].split("\\")[-1].split('.')[0] + '_' + row["message_id"]
        if body == 'ProcessErrorOccurred':
            continue
        rec = Recording.from_json(chldr_sh.col_settings, ORIGINAL_SAMPLE_RATE)
        rec.SetSignalHandlers(body,0,0,0)
            
    idx = rec.GetSpecificSignalChannelIndex(ch_to_use)
    signal = rec.SignalHandler[idx].GetBandpassFilteredData(ORIGINAL_SAMPLE_RATE,60,1,order_lp=3,order_hp=3)
    
    ds_sig, ds_fids = downsample_data(ORIGINAL_SAMPLE_RATE, TARGET_SAMPLE_RATE, signal, manual_fiducials)
    
    masks, masks_names = convert_fiducials_to_masks(ds_sig, ds_fids)
    
    segment_samples = SEGMENT_LENGTH * TARGET_SAMPLE_RATE
    overlap_samples= OVERLAP_LENGTH * TARGET_SAMPLE_RATE
    
    iteration = 0
    for start_idx in range(0,len(ds_sig), segment_samples - overlap_samples):
        iteration = iteration + 1
        end_idx = start_idx + segment_samples
        if end_idx > len(ds_sig):
            start_idx = len(ds_sig)-segment_samples
            end_idx = len(ds_sig)
        
        mask = ds_fids.apply(lambda row: row.between(start_idx, end_idx).all(), axis=1)
        fiducial_matrix = ds_fids[mask].values
        fiducial_matrix = fiducial_matrix - start_idx
        
        sig_segment = ds_sig[start_idx:end_idx]
        sig_segment = SCALER.fit_transform(sig_segment.reshape(-1,1)).squeeze()
        masks_segment = masks[start_idx:end_idx,:]
        
        masks_segment = ensure_full_beat_start(masks_segment)
        
        with h5py.File(TARGET_H5, 'a') as h5f:
            h5f.create_dataset(f"{name}-{iteration}/input", data=sig_segment)
            h5f.create_dataset(f"{name}-{iteration}/masks", data=masks_segment)
            