from os import mkdir
import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import gzip


def un_gz(file_name:str, dir:str):
    """
    Unzips a .gz file to a specified directory.

    Args:
        file_name (str): The path to the .gz file.
        dir (str): The directory where the unzipped file will be stored.
    """
    f_name = file_name.split('/')[-1].replace(".gz", "")
    os.makedirs(dir, exist_ok=True)  # Create the directory if it doesn't exist.
    f_name = dir + '/' + f_name
    with gzip.GzipFile(file_name, 'rb') as g_file:  # Use context manager for file handling
        with open(f_name, "wb") as out_file:
            out_file.write(g_file.read())


def get_patient_list(dir_wsi:str, dir_rad:str):
    """
    Identifies matching patients and their corresponding slide IDs from WSI and radiology data directories.

    Args:
        dir_wsi (str): Directory containing WSI (Whole Slide Image) H5 files. Assumes a 'h5_files' subdirectory.
        dir_rad (str): Directory containing processed radiology data.  Expects patient directories directly within.

    Returns:
        tuple: A tuple containing two lists:
            - patient_list (list): A list of patient IDs found in both WSI and radiology datasets.
            - slide_list (list): A list of corresponding slide IDs for the matched patients.
    """
    wsi_list = glob.glob(dir_wsi + '/h5_files/*.h5')
    rad_list = glob.glob(dir_rad + '/*')

    # Extract slide IDs from WSI file paths.
    wsi_list = [wsi.split('/')[-1][:-3] for wsi in wsi_list]

    # Extract patient IDs from WSI slide IDs.  Assumes patient ID is the first 12 characters.
    wsi_patient_list = [wsi[:12] for wsi in wsi_list]

    # Extract patient IDs from radiology directory names.
    rad_patient_list = [rad.split('/')[-1] for rad in rad_list]  # More descriptive variable name

    # Find the intersection of patient IDs present in both WSI and radiology data.
    patient_list = list(set(wsi_patient_list) & set(rad_patient_list))

    # Get the corresponding slide IDs for the matched patients.
    slide_list = []
    for patient in patient_list:
        try:
            slide_list.append(wsi_list[wsi_patient_list.index(patient)]) #Potentially problematic if patient appears more than once in WSI list with different slides.
        except ValueError:
            print(f"Warning: Patient {patient} found in rad_list but not uniquely in wsi_patient_list. Skipping.") #Added error handling for cases where patient ID isn't found
            continue #Skips to the next patient to avoid crash and continue processing other patients.

    return patient_list, slide_list


def get_rad(dir, root_dir):
    """
    Unzips and organizes radiology data for each patient from a source directory to a target directory.

    This function iterates through patient directories within the input directory, finds the earliest
    radiology data (based on directory name date), and unzips the relevant brainless file.

    Args:
        dir (str): The root directory containing patient-specific subdirectories of radiology data.
        root_dir (str): The root directory where the unzipped radiology data will be stored, organized by patient.
    """
    patient_list = glob.glob(dir + '/*')
    format_pattern = '%m-%d-%Y'

    for rad in patient_list:
        patient_name = rad.split('/')[-1]
        rad_contents = glob.glob(rad + '/*')  # Get all contents within the patient's directory
        rad_dirs = [r for r in rad_contents if os.path.isdir(r)]  # Filter out only the directories

        if not rad_dirs:
            continue  # Skip if no directories are found

        if len(rad_dirs) > 1:
            # Choose the directory with the earliest date.
            time_list = [r.split('/')[-1][:10] for r in rad_dirs]
            try:
                time_list = [datetime.strptime(time, format_pattern) for time in time_list]
            except ValueError as e:
                 print(f"Skipping {rad} due to date format error: {e}")
                 continue #Skip to next patient if date format issue arises

            sorted_ids = sorted(range(len(time_list)), key=lambda k: time_list[k])
            rad_dir = rad_dirs[sorted_ids[0]]
        else:
            rad_dir = rad_dirs[0]

        # Construct the path to the brain lesion file.
        brain_les_rel_path = rad_dir.split('/')[-1] + '_brainles/normalized_bet/' + rad_dir.split('/')[-1] + '_t1c_bet_normalized.nii.gz'
        brain_les = os.path.join(rad_dir, brain_les_rel_path)

        if not os.path.exists(brain_les):
            print(f"Warning: Brain lesion file not found: {brain_les}")
            continue

        un_gz(brain_les, os.path.join(root_dir, patient_name))


def generate_survival_interval(survival_days, n_bin):
    """
    Generates survival intervals by dividing survival days into bins based on quantiles.

    Args:
        survival_days (list): A list of survival times.
        n_bin (int): The number of bins to divide the survival times into.

    Returns:
        list: A list of integers, where each integer represents the bin index for the corresponding survival day.
    """
    quartiles = np.quantile(survival_days, [1/n_bin*i for i in range(1, n_bin+1)])
    survival_interval = []
    for day in survival_days:
        for i in range(n_bin):
            if day <= quartiles[i]:
                survival_interval.append(i)
                break # Found bin, no need to continue inner loop

    return survival_interval


def main():
    Datasets = ['TCGA-GBM', 'TCGA-LGG']
    data_root1 = '/data1/WSI/Pathology_Radiology/Dataset/'
    data_root2 = '/data1/yhchen/'

    for dataset in Datasets:
        # get_rad(data_root1+dataset+'/Rad_nii_processed',data_root2+dataset) # only running this once.
        patient_list, slide_list = get_patient_list(data_root1+dataset+'/patch_512/conch_512_con', data_root2+dataset+'/Radiology')
        clinical_info = pd.read_excel(data_root2+dataset+f'/{dataset}_Clinical.xlsx')

        censor, survival_days, survival_interval = [], [], []
        for patient in patient_list:
            patient_info = clinical_info[clinical_info['bcr_patient_barcode'] == patient]
            if patient_info.shape[0] == 0:
                raise ValueError(f'Patient not found in clinical info: {patient}')

            vital_status = patient_info['vital_status'].values[0]
            if vital_status == 'Alive':
                censor.append(1)
                survival_days.append(patient_info['days_to_last_followup'].values[0])
            elif vital_status == 'Dead':
                censor.append(0)
                survival_days.append(patient_info['days_to_death'].values[0])
            else:
                raise ValueError(f'Patient status not found: {patient}')

        survival_interval = generate_survival_interval(survival_days, 4)
        survival_info = pd.DataFrame({
            'case_id': patient_list,
            'slide_id': slide_list,
            'censor': censor,
            'survival_days': survival_days,
            'survival_interval': survival_interval
        })
        survival_info.to_csv(data_root2+dataset+f'/{dataset}_survival_info.csv', index=False)


if __name__ == '__main__':
    main()
