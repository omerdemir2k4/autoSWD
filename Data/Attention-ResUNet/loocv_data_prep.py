import numpy as np
import os
import mne
import pandas as pd
from datetime import time

from datetime import time

def binary_extraction_100hz(edf_file_path, excel_file_path, num_channel):
    seizure_data = pd.read_excel(excel_file_path, header=None, 
                                 converters={0: str, 1: str})

    # The format strings for the different time formats, such as
    time_format_1 = '%H:%M:%S.%f'  # For '08:42:54.3'
    time_format_2 = '%M:%S.%f'      # For '43:43,0'
    time_format_3 = '%H:%M:%S'      # For '08:43:43'

    # A helper function to parse time strings with different formats
    def parse_time_string(time_str):
        if not isinstance(time_str, str):
            # If it's already a time object, return it
            return time_str
        
        time_str = time_str.replace(',', '.')

        try:
            return pd.to_datetime(time_str, format=time_format_1).time()
        except ValueError:
            try:
                return pd.to_datetime(time_str, format=time_format_2).time()
            except ValueError:
                try:
                    return pd.to_datetime(time_str, format=time_format_3).time()
                except ValueError:
                    # Fallback for unexpected formats
                    print(f"Could not parse time string: {time_str}")
                    return None

    seizure_data[0] = seizure_data[0].apply(parse_time_string)
    seizure_data[1] = seizure_data[1].apply(parse_time_string)
    
    seizure_data.dropna(subset=[0, 1], inplace=True)

    seizure_data = seizure_data[
        (seizure_data[0] >= time(9, 0, 0)) &
        (seizure_data[1] <= time(12, 0, 0))
    ]

    sampling_rate = 100

    def time_to_milliseconds(time_entry):
        if isinstance(time_entry, time):
            milliseconds = (time_entry.hour * 3600 + time_entry.minute * 60 + time_entry.second) * 1000 + int(time_entry.microsecond / 1000)
        else:
            t = pd.to_datetime(time_entry)
            milliseconds = (t.hour * 3600 + t.minute * 60 + t.second) * 1000 + int(t.microsecond / 1000)
        return milliseconds

    file = mne.io.read_raw_edf(edf_file_path, preload=True)
    file.resample(sampling_rate)

    start_time_of_recording = file.info['meas_date']
    start_time_ms = time_to_milliseconds(start_time_of_recording)
    channel_array = file.get_data(picks=num_channel - 1, units="mV")[0]

    start_9 = time_to_milliseconds(time(9, 0, 0)) - start_time_ms
    end_12 = time_to_milliseconds(time(12, 0, 0)) - start_time_ms

    start_idx = int(start_9 * sampling_rate // 1000)
    end_idx = int(end_12 * sampling_rate // 1000)

    channel_array = channel_array[start_idx:end_idx]
    channel_array = (channel_array - np.mean(channel_array)) / np.std(channel_array)

    binary = np.zeros(len(channel_array), dtype=np.int8)

    # The logic here is already correct for the filtered DataFrame
    seizure_data['start_time_ms'] = seizure_data[0].apply(time_to_milliseconds) - start_time_ms - start_9
    seizure_data['end_time_ms'] = seizure_data[1].apply(time_to_milliseconds) - start_time_ms - start_9

    for _, row in seizure_data.iterrows():
        s = int(max(0, row['start_time_ms'] * sampling_rate // 1000))
        e = int(min(len(channel_array), row['end_time_ms'] * sampling_rate // 1000))
        binary[s:e] = 1

    return binary, channel_array

basal_2 = r"GA-D#9,10,11,12 BASAL RECORDING 220824.edf"
basal_4 = r"GA-KOL#18,19,20,21 BASAL RECORDING 151024.edf"
basal_5 = r"GA-KOL#30,31,32,33 BASAL RECORDİNG (0-3h) 220425.edf"
enj_1 = r"GA-KI-D#5,6,7,8 1 MGKG ATROPIN İP ENJ 090827.edf"
sf_5 = r"GA-KOL# 23-24-25 SF ENJ - GA-CHEM#17 BASAL.edf"

recordings = [
    {"edf_path": basal_2, "excel_path": r"cleaned_ga-kol-6_bazal.xlsx", "channel": 3},
    {"edf_path": basal_2, "excel_path": r"cleaned_ga-kol-7_bazal.xlsx", "channel": 5},
    {"edf_path": basal_2, "excel_path": r"cleaned_ga-kol-8_bazal.xlsx", "channel": 8},
    {"edf_path": basal_4, "excel_path": r"cleaned_ga-kol-18_bazal.xlsx", "channel": 1},
    {"edf_path": basal_4, "excel_path": r"cleaned_ga-kol-19_bazal.xlsx", "channel": 3},
    {"edf_path": enj_1, "excel_path": r"cleaned_ga-kol-1_enj.xlsx", "channel": 2},
    {"edf_path": enj_1, "excel_path": r"cleaned_ga-kol-2_enj.xlsx", "channel": 4},
    {"edf_path": enj_1, "excel_path": r"cleaned_ga-kol-3_enj.xlsx", "channel": 5},
    {"edf_path": enj_1, "excel_path": r"cleaned_ga-kol-4_enj.xlsx", "channel": 8},
    {"edf_path": sf_5, "excel_path": r"cleaned_ga-chem-17_bazal.xlsx", "channel": 1},
    {"edf_path": basal_5, "excel_path": r"Ccleaned_ga-kol-30_bazal.xlsx", "channel": 1},
    {"edf_path": basal_5, "excel_path": r"cleaned_ga-kol-31_bazal.xlsx", "channel": 3},
    {"edf_path": basal_5, "excel_path": r"cleaned_ga-kol-32_bazal.xlsx", "channel": 5}
]

save_dir = "loocv_raw_segments_9_12_13rec"
os.makedirs(save_dir, exist_ok=True)

for i, rec in enumerate(recordings):
    binary, signal = binary_extraction_100hz(rec["edf_path"], rec["excel_path"], rec["channel"])
    name = os.path.splitext(os.path.basename(rec["excel_path"]))[0]

    X = signal.reshape(-1, 1)
    Y = binary.reshape(-1, 1)

    np.save(os.path.join(save_dir, f"X_{name}.npy"), X)
    np.save(os.path.join(save_dir, f"Y_{name}.npy"), Y)
    print(f"Saved {name}: shape={X.shape}")
