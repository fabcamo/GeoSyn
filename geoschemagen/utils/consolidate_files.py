import os
import shutil
import random
import sys


def print_progress_bar(current: int, total: int, bar_length: int = 40) -> None:
    """Prints a progress bar to track the completion of a task.

    Args:
        current (int): The current progress count.
        total (int): The total number of items to process.
        bar_length (int, optional): The length of the progress bar. Defaults to 40.
    """
    # Calculate the progress percentage
    progress = current / total
    # Create the progress bar
    block = int(bar_length * progress)
    bar = "#" * block + "-" * (bar_length - block)
    # Print the progress bar
    sys.stdout.write(f"\rProgress: [{bar}] {current}/{total} files")
    # Flush the output to the console to show the progress bar in the same line
    sys.stdout.flush()


def consolidate_files_in_batches(
    folders: list[str],
    file_format: str,
    output_folder: str,
    batch_size: int = 1000,
) -> None:
    """Consolidates files from multiple folders into a single folder with randomized order.

    The function processes the files in batches to avoid memory issues, shuffles them
    to randomize their order, and renames them sequentially in the format `cs_X`.

    Args:
        folders (list[str]): List of folder paths to process.
        file_format (str): File format to search for (e.g., '.jpeg').
        output_folder (str): Path to the output folder where files will be saved.
        batch_size (int, optional): Number of files to process in a single batch.
            Defaults to 1000.

    Raises:
        OSError: If an error occurs during file copying.
    """
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
            # Get the full path of each file in the folder
            for file_name in os.listdir(folder)
            # Filter files by format (case-insensitive)
            if file_name.lower().endswith(file_format.lower())
        )

    # Check if any files were found
    if not files:
        print("No matching files found.")
        return

    # Shuffle the entire list of files
    random.shuffle(files)

    # Process files in batches to avoid memory issues
    total_files = len(files)
    global_index = 0
    print(f"Total files to process: {total_files}")

    # Start the progress bar and process files in batches
    for i in range(0, total_files, batch_size):
        batch = files[i:i + batch_size]
        random.shuffle(batch)

        # Process each file in the batch
        for file_path in batch:
            new_file_name = f"cs_{global_index}{file_format}"
            new_file_path = os.path.join(output_folder, new_file_name)
            try:
                shutil.copy(file_path, new_file_path)
                global_index += 1
                print_progress_bar(global_index, total_files)
            except OSError as e:
                print(f"\nError copying {file_path}: {e}")

    # Finish the progress bar
    sys.stdout.write("\n")
    print("File consolidation complete.")


# Example usage
if __name__ == "__main__":
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
