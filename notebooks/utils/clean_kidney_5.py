import os
import glob

# Set the directory where the files are located
folder_path = '/home/mithil/PycharmProjects/SenNetKideny/data/train/50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2_'  # Replace with the path to your folder

# Get all the .jp2 files in the folder
jp2_files = glob.glob(os.path.join(folder_path, '*.jp2'))

# Sort the files to maintain order
jp2_files.sort()

# Get the total number of files for zero padding
total_files = len(jp2_files)
max_length = len(str(total_files))

# Rename each .jp2 file to .tif with sequential numbering
for i, file_path in enumerate(jp2_files):
    # Construct the new file name with proper zero padding and .tif extension
    new_file_name = str(i).zfill(max_length) + '.tif'
    new_file_path = os.path.join(folder_path, new_file_name)

    # Rename the file
    os.rename(file_path, new_file_path)
    print(f'Renamed {file_path} to {new_file_path}')
