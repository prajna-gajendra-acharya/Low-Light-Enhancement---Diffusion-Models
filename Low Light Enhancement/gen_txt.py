import os
import glob

# Configuration
data_root = "/content/drive/MyDrive/CV_Project/LightenDiffusion/LOL-v1"
train_folder = 'train'  # Training data folder
val_folder = 'val'    # Validation data folder

print(f"\nFiles in {val_folder}/low:")
print(sorted(os.listdir(os.path.join(data_root, val_folder, 'low'))))
print(f"\nFiles in {val_folder}/high:")
print(sorted(os.listdir(os.path.join(data_root, val_folder, 'high'))))

def generate_file_list(folder_name, output_filename):
    # distinct paths for low and high images
    low_path = os.path.join(data_root, folder_name, 'low')
    high_path = os.path.join(data_root, folder_name, 'high')

    # Check if paths exist
    if not os.path.exists(low_path) or not os.path.exists(high_path):
        print(f"Error: Could not find directories in {folder_name}")
        return

    # Get all low light images
    low_images = sorted(glob.glob(os.path.join(low_path, '*.png')))
    lines = []

    print(f"Processing {folder_name}...")

    for low_img_path in low_images:
        # Get filename (e.g., '1.png')
        filename = os.path.basename(low_img_path)

        # Construct path to corresponding high light image
        high_img_path = os.path.join(high_path, filename)

        # Verify the pair exists
        if os.path.exists(high_img_path):
            # Write relative paths exactly as dataset.py expects them
            lines.append(f"{low_img_path} {high_img_path}")
        else:
            print(f"Warning: No high-light pair found for {filename}")

    # Write to .txt file in the data root
    output_path = os.path.join(data_root, output_filename)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Success: Generated {output_path} with {len(lines)} image pairs.")

# Generate the lists
generate_file_list(train_folder, 'LOLv1_train.txt')
generate_file_list(val_folder, 'LOLv1_val.txt')
