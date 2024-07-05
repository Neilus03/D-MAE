import os
import numpy as np

def decompress_npz_files(input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.npz'):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, file.replace('.npz', '.npy'))

                # Load the .npz file
                with np.load(input_file_path) as data:
                    # Extract the data and save it as .npy
                    np.save(output_file_path, data['data'])

                print(f"Decompressed {input_file_path} to {output_file_path}")

if __name__ == "__main__":
    input_directory_train = "/home/ndelafuente/Desktop/D-MAE/data/data/train/compressed_depth"
    output_directory_train = "/home/ndelafuente/Desktop/D-MAE/data/data/train/decompressed_depth"
    decompress_npz_files(input_directory_train, output_directory_train)

    input_directory_val = "/home/ndelafuente/Desktop/D-MAE/data/data/val/compressed_depth"
    output_directory_val = "/home/ndelafuente/Desktop/D-MAE/data/data/val/decompressed_depth"
    decompress_npz_files(input_directory_val, output_directory_val)
