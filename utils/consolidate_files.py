import os
import shutil
import random
import sys
import time

def print_progress_bar(current, total, bar_length=40):
    progress = current / total
    block = int(bar_length * progress)
    bar = "#" * block + "-" * (bar_length - block)
    sys.stdout.write(f"\rProgress: [{bar}] {current}/{total} files")
    sys.stdout.flush()

def consolidate_files_in_batches(folders, file_format, output_folder, batch_size=1000):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Gather all files with the specified format from all folders
    files = []
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Error: Folder does not exist: {folder}")
            continue
        files.extend(
            os.path.join(folder, file_name)
            for file_name in os.listdir(folder)
            if file_name.lower().endswith(file_format.lower())
        )

    if not files:
        print("No matching files found.")
        return

    # Shuffle the entire list of files
    random.shuffle(files)

    # Process files in batches
    total_files = len(files)
    global_index = 0
    print(f"Total files to process: {total_files}")

    for i in range(0, total_files, batch_size):
        batch = files[i:i + batch_size]
        random.shuffle(batch)

        for file_path in batch:
            new_file_name = f"cs_{global_index}{file_format}"
            new_file_path = os.path.join(output_folder, new_file_name)
            try:
                shutil.copy(file_path, new_file_path)
                global_index += 1
                print_progress_bar(global_index, total_files)
            except Exception as e:
                print(f"\nError copying {file_path}: {e}")

    # Finish the progress bar
    sys.stdout.write("\n")
    print("File consolidation complete.")

# Example usage
folders = [
    r'D:\GeoSchemaGen\tests\typeA_RFTrue_20241209',
    r'D:\GeoSchemaGen\tests\typeB_RFTrue_20241209',
    r'D:\GeoSchemaGen\tests\typeC_RFTrue_20241209',
    r'D:\GeoSchemaGen\tests\typeD_RFTrue_20241209',
    r'D:\GeoSchemaGen\tests\typeE_RFTrue_20241209',
    r'D:\GeoSchemaGen\tests\typeF_RFTrue_20241209',
]
output_folder = r'D:\GeoSchemaGen\tests\cs_12092024'
file_format = ".png"
batch_size = 1000  # Adjust batch size as needed

consolidate_files_in_batches(folders, file_format, output_folder, batch_size)
