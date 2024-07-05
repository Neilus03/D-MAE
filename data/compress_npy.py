import os
import numpy as np

def compress_npy_files(input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.npy'):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, file.replace('.npy', '.npz'))

                # Load the .npy file
                data = np.load(input_file_path)

                # Save the data in .npz format
                np.savez_compressed(output_file_path, data=data)

                print(f"Compressed {input_file_path} to {output_file_path}")

if __name__ == "__main__":
    input_directory = "/home/ndelafuente/Desktop/D-MAE/data/data/train/depth" 
    output_directory = "/home/ndelafuente/Desktop/D-MAE/data/data/train/compressed_depth"
    compress_npy_files(input_directory, output_directory)
    input_directory = "/home/ndelafuente/Desktop/D-MAE/data/data/val/depth"
    output_directory = "/home/ndelafuente/Desktop/D-MAE/data/data/val/compressed_depth"