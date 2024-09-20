from datasets.carla import *
from tools.geometry import *
from tools.utils import *
import argparse

from numpy.linalg import inv
import random
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def find_center(array):
    indices = np.argwhere(array == 1)

    min_x = np.min(indices[:, 1])
    max_x = np.max(indices[:, 1])
    min_y = np.min(indices[:, 0])
    max_y = np.max(indices[:, 0])

    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2

    return center_x, center_y


def find_total_height(array):
    indices = np.where(array == 1)

    min_y = np.min(indices[0])
    max_y = np.max(indices[0])

    total_height = max_y - min_y + 1
    return total_height


def resize_image_by_height(image, new_height):
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(aspect_ratio * new_height)

    resized_image = image.resize((new_width, int(new_height)))
    return resized_image


def augment():
    animals = [
        "bear",
        "cow",
        "horse",
        "deer",
        "donkey",
        "elk",
        "fox",
        "lion",
        "wolf",
        "moose"
    ]

    sizes = [
        [1.1, 2.75, 1.75],
        [1.1, 2.25, 1.25],
        [1.1, 2.5, 3],
        [1.1, 2.3, 1.5],
        [1.25, 2.5, 1.5],
        [1.6, 2.6, 1.7],
        [1, 2, 1.5],
        [1.5, 2.5, 1.6],
        [1.25, 2, 1.4],
        [1.6, 2.6, 1.7],
    ]

    cameras = [
        'left_front_camera',
        'front_camera',
        'right_front_camera',
        'left_back_camera',
        'back_camera',
        'right_back_camera'
    ]

    carla_data = CarlaDataset(args.data_dir, False)
    carla_data.return_info = True

    for images, intrinsics, extrinsics, labels, oods, info in tqdm(carla_data):
        save = info['index']
        path = os.path.join(args.save_dir, f"agents/{info['agent_number']}/")

        for camera in cameras:
            os.makedirs(os.path.join(path, camera), exist_ok=True)
        os.makedirs(os.path.join(path, 'bev_semantic'), exist_ok=True)

        bev_seg = cv2.imread(os.path.join(info['agent_path'] + "bev_semantic", f'{info["index"]}.png'))
        cams = [Image.open(os.path.join(info['agent_path'] + cn, f'{info["index"]}.png')) for cn in cameras]

        excluded = labels[3].numpy()
        n_obj = random.randint(1, 4)
        failed = 0

        while n_obj > 0:
            a = random.randrange(0, len(animals))
            file = random.choice(os.listdir(f"{pseudo_path}/{animals[a]}/"))
            ood = Image.open(f"{pseudo_path}/{animals[a]}/{file}")

            size = sizes[a]
            rot = euler_to_quaternion(0, 0, 0)
            trans = [
                random.randint(-30, 30),
                random.randint(-30, 30),
                size[2] / 2
            ]

            bev_box = draw_bev(trans, rot, size).astype(bool)

            if np.sum(excluded[bev_box]) > 0:
                if failed >= 15:
                    break
                failed += 1
                continue

            bev_seg[bev_box == 1, :] = 0
            excluded[bev_box] = 1

            for ci, cn in enumerate(cameras):
                I = intrinsics[ci]
                E = np.linalg.inv(extrinsics[ci])

                cam_box = draw_cam(
                    trans, rot, size, I, E,
                    dataset='carla',
                )

                if np.sum(cam_box) > 0:
                    indices = np.where(cam_box == 1)
                    min_y = np.min(indices[0])
                    max_y = np.max(indices[0])
                    min_x = np.min(indices[1])
                    max_x = np.max(indices[1])

                    r_ood = ood.resize((max_x - min_x + 1, max_y - min_y + 1))
                    cams[ci].paste(r_ood, (min_x, min_y), r_ood)

            n_obj -= 1

        cv2.imwrite(os.path.join(path + "bev_semantic", f'{save}.png'), bev_seg)
        for i, cam in enumerate(cams):
            cam.save(os.path.join(path + cameras[i], f'{save}.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir")
    parser.add_argument("save_dir")

    args = parser.parse_args()
    pseudo_path = './outputs/pseudo_f'

    augment()
