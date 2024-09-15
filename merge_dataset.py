import os
import shutil
from tqdm import tqdm


def merge_datasets(set_a, set_b, output_dir):
    for root, _, files in tqdm(os.walk(set_a)):
        rel_path = os.path.relpath(root, set_a)
        output_folder_path = os.path.join(output_dir, rel_path)

        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        for file_name in files:
            if file_name.endswith('.png'):
                shutil.copy(
                    os.path.join(root, file_name),
                    os.path.join(output_folder_path, file_name)
                )

    # Copy and rename files from train_aug_more to output_dir
    for root, _, files in tqdm(os.walk(set_b)):
        rel_path = os.path.relpath(root, set_b)
        output_folder_path = os.path.join(output_dir, rel_path)

        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        # Get the maximum idx in the corresponding output folder
        max_idx = max(
            [int(f.split('.')[0]) for f in os.listdir(output_folder_path) if f.endswith('.png')],
            default=-1
        )

        for file_name in files:
            if file_name.endswith('.png'):
                old_idx = int(file_name.split('.')[0])
                new_idx = max_idx + old_idx + 1
                new_file_name = f"{new_idx}.png"
                shutil.copy(
                    os.path.join(root, file_name),
                    os.path.join(output_folder_path, new_file_name)
                )


set_a = "/home/bny220000/data/projects/data/carla/test"
set_b = "/home/bny220000/data/projects/data/carla/ood_more"
output_dir = "/home/bny220000/data/projects/data/carla/ood_test_more"
merge_datasets(set_a, set_b, output_dir)
