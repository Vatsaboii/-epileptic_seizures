import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.io import read_raw_edf

def save_images_from_edf(file_path, save_folder, is_seizure):
    raw = read_raw_edf(file_path, preload=True, verbose=False)
    os.makedirs(save_folder, exist_ok=True)
    for i, channel_name in enumerate(raw.ch_names):
        plt.figure(figsize=(10, 4))
        plt.plot(raw.times, raw.get_data()[i], color='b')
        plt.title(f"Channel: {channel_name} - {'Seizure' if is_seizure else 'Non-Seizure'}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        file_suffix = "seizure" if is_seizure else "non_seizure"
        save_path = os.path.join(save_folder, f"{channel_name}_{file_suffix}.png")
        plt.savefig(save_path)
        plt.close()

def process_edf_files(edf_folder, output_folder):
    for file_name in os.listdir(edf_folder):
        if file_name.endswith(".edf"):
            base_name = file_name.replace('.edf', '')
            seizure_file = base_name + '.edf.seizures'
            is_seizure = os.path.isfile(os.path.join(edf_folder, seizure_file))
            folder_name = 'seizure' if is_seizure else 'non_seizure'
            save_folder = os.path.join(output_folder, folder_name, base_name)
            
            file_path = os.path.join(edf_folder, file_name)
            save_images_from_edf(file_path, save_folder, is_seizure)
edf_folder = "/Users/srivatsapalepu/seizure/Seizures"
output_folder = "/Users/srivatsapalepu/seizure/output_images"
process_edf_files(edf_folder, output_folder)
