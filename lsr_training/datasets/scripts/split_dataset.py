import os
import random
import shutil

def split_dataset(base_dir, valid_count, test_count):
    down_scales = [2, 3, 4]  # fixed
    source_folders = ['HR']
    source_folders.extend([f'LR/X{scale}' for scale in down_scales])
    latent_folders = ['HR_sdxl_latent']
    latent_folders.extend([f'LR_sdxl_latent/X{scale}' for scale in down_scales])
    
    # Define paths
    hr_source_path = os.path.join(base_dir, 'HR')
    
    # Get all HR images
    try:
        hr_images = [f.split('.')[0] for f in os.listdir(hr_source_path) if os.path.isfile(os.path.join(hr_source_path, f))]
    except FileNotFoundError:
        print(f"Error: Directory not found at {hr_source_path}")
        return

    # Shuffle the images for random sampling
    random.shuffle(hr_images)

    # Sample validation and test sets
    valid_set = hr_images[:valid_count]
    test_set = hr_images[valid_count:valid_count + test_count]

    sets = {
        'valid': valid_set,
        'test': test_set
    }

    for set_name, image_list in sets.items():
        target_base_path = os.path.join(base_dir, set_name)
        os.makedirs(target_base_path, exist_ok=True)

        for image_name in image_list:
            for folder in source_folders:
                source_file_path = os.path.join(base_dir, folder, f'{image_name}.jpg')
                target_file_path = os.path.join(target_base_path, f'{image_name}.jpg')

                if folder == 'HR':
                    if os.path.exists(source_file_path):
                        shutil.move(source_file_path, target_file_path)
                        print(f"Moved {source_file_path} to {target_file_path}")
                    else:
                        print(f"Warning: Source file not found at {source_file_path}")
                else:
                    if os.path.exists(source_file_path):
                        os.remove(source_file_path)
                        print(f"Removed {source_file_path}")
                    else:
                        print(f"Warning: Source file not found at {source_file_path}")

            for folder in latent_folders:
                source_file_path = os.path.join(base_dir, folder, f'{image_name}.npy')

                if os.path.exists(source_file_path):
                    os.remove(source_file_path)
                    print(f"Removed {source_file_path}")
                else:
                    print(f"Warning: Source file not found at {source_file_path}")

    print("Dataset splitting complete.")

if __name__ == '__main__':
    dataset_base_dir = '/home/m11215122/datasets/OpenImages'
    num_validation = 400
    num_test = 1000
    split_dataset(dataset_base_dir, num_validation, num_test)