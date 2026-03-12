from DataPreparationFunctions.identify_data_cursor_file_pairs import identify_data_cursor_file_pairs
excel_destination = 'C:/Users/ZD94OW/SeismicHeartLocal/Code/BeatToBeatAlgo/data_to_train.xlsx'
cursor_root = 'C:/Users/ZD94OW/OneDrive - Aalborg Universitet/SeismicHeart/Beat to Beat Fiducials/Cursors/CursorFolder/'
data_root = 'C:/Users/ZD94OW/SeismicHeartLocal/Data/Raw Data/'
a=identify_data_cursor_file_pairs(excel_destination, cursor_root, data_root)
    