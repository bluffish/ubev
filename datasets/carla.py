import json
import math
import os
import random

import torchvision
from torch.utils.data import Subset
from torchvision.transforms.functional import center_crop

from tools.geometry import *

def split_path_into_folders(path):
    folders = []
    while True:
        path, folder = os.path.split(path)
        if folder:
            folders.append(folder)
        else:
            if path:
                folders.append(path)
            break
    folders.reverse()
    return folders

def get_intrinsics(image_size_x, image_size_y, fov):
    intrinsics = np.identity(3)
    intrinsics[0, 2] = image_size_x / 2.0
    intrinsics[1, 2] = image_size_y / 2.0
    intrinsics[0, 0] = intrinsics[1, 1] = image_size_x / (
            2.0 * np.tan(fov * np.pi / 360.0))

    return intrinsics

def nn_resample(img, shape):
    def per_axis(in_sz, out_sz):
        ratio = 0.5 * in_sz / out_sz
        return np.round(np.linspace(ratio - 0.5, in_sz - ratio - 0.5, num=out_sz)).astype(int)

    return img[per_axis(img.shape[0], shape[0])[:, None],
               per_axis(img.shape[1], shape[1])]

class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train, pos_class, pseudo=False, map_uncertainty=False):
        self.is_train = is_train
        self.return_info = False
        self.pos_class = pos_class
        self.pseudo = pseudo
        self.map_uncertainty = map_uncertainty

        self.data_path = data_path

        self.mode = 'train' if self.is_train else 'val'

        self.data = []

        agent_folders = []
        for dirpath, dirnames, filenames in os.walk(self.data_path):
            if 'sensors.json' in filenames:
                agent_folders.append(dirpath)

        for agent_path in agent_folders:
            if map_uncertainty and not os.path.exists(os.path.join(agent_path, 'bev_mapping')):
                continue
            splitted_path = split_path_into_folders(agent_path)
            agent_id = splitted_path[-1]
            town_name = splitted_path[-3]
            for filename in os.listdir(os.path.join(agent_path, 'front_camera')):
                frame = int(filename.split('.png')[0])
                self.data.append((agent_path, agent_id, frame))

        self.offset = 0

        self.to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )

        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution, bev_start_position, bev_dimension
        )

    def get_input_data(self, index, agent_path):
        images = []
        intrinsics = []
        extrinsics = []

        with open(os.path.join(agent_path, 'sensors.json'), 'r') as f:
            sensors = json.load(f)

        for sensor_name, sensor_info in sensors['sensors'].items():
            if sensor_info["sensor_type"] == "sensor.camera.rgb" and sensor_name != "birds_view_camera":
                image = Image.open(os.path.join(agent_path, sensor_name, f'{index}.png'))
                if image.size != (480, 224):
                    # 1600*900 -> 1600*747 -> 480*224
                    orig_w, orig_h = image.size
                    assert (orig_w / orig_h) <= (480 / 224) # Only when we didn't change horizontal FOV we can calculate intrinsic correctly.
                    image = center_crop(image, (480 / orig_w * orig_h, orig_w))
                    image =  image.resize((480, 224), Image.Resampling.BILINEAR)

                # intrinsic = torch.tensor(sensor_info["intrinsic"])
                sensor_options = sensor_info['sensor_options'] # width and height are in sensor_options['image_size_x'] and sensor_options['image_size_y']
                
                intrinsic = get_intrinsics(480, 224, sensor_options["fov"]).astype(np.float32)
                translation = np.array(sensor_info["transform"]["location"])
                rotation = sensor_info["transform"]["rotation"]

                rotation[0] += 90
                rotation[2] -= 90

                r = Rotation.from_euler('zyx', rotation, degrees=True)

                extrinsic = np.eye(4, dtype=np.float32)
                extrinsic[:3, :3] = r.as_matrix()
                extrinsic[:3, 3] = translation
                extrinsic = np.linalg.inv(extrinsic)

                normalized_image = self.to_tensor(image)

                images.append(normalized_image)
                intrinsics.append(torch.tensor(intrinsic))
                extrinsics.append(torch.tensor(extrinsic))
                image.close()

        images, intrinsics, extrinsics = (torch.stack(images, dim=0),
                                          torch.stack(intrinsics, dim=0),
                                          torch.stack(extrinsics, dim=0))

        return images, intrinsics, extrinsics

    def get_label(self, index, agent_path):
        # label_r = Image.open(os.path.join(agent_path, "bev_semantic", f'{index}.png'))
        label_r = Image.open(os.path.join(agent_path, "birds_view_semantic_camera", f'{index}.png'))
        if label_r.size != (200, 200):
            label_r =  label_r.resize((200, 200), Image.Resampling.NEAREST)

        label = np.array(label_r)
        label_r.close()

        if self.map_uncertainty:
            mapped_epistemic = np.load(os.path.join(agent_path, "bev_mapping_epistemic", f'{index}.npy'))
            if mapped_epistemic.shape != (200, 200):
                mapped_epistemic =  nn_resample(mapped_epistemic, (200, 200))
        else:
            mapped_epistemic = None

        empty = np.ones(self.bev_dimension[:2])

        road = mask(label, (128, 64, 128))
        lane = mask(label, (157, 234, 50))
        vehicles = mask(label, (0, 0, 142))

        if np.sum(vehicles) < 5:
            lane = mask(label, (50, 234, 157))
            vehicles = mask(label, (142, 0, 0))

        # ood = mask(label, (0, 0, 0))
        ood = mask(label, (50, 100, 144))
        bounding_boxes = find_bounding_boxes(ood)
        ood = draw_bounding_boxes(bounding_boxes)

        if self.pos_class == 'vehicle':
            empty[vehicles == 1] = 0
            label = np.stack((vehicles, empty))

        elif self.pos_class == 'road':
            road[lane == 1] = 1
            road[vehicles == 1] = 1

            # this is a refinement step to remove some impurities in the label caused by small objects
            road = (road * 255).astype(np.uint8)
            kernel_size = 2

            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            road = cv2.dilate(road, kernel, iterations=1)
            road = cv2.erode(road, kernel, iterations=1)
            empty[road == 1] = 0

            label = np.stack((road, empty))
        elif self.pos_class == 'lane':
            empty[lane == 1] = 0

            label = np.stack((lane, empty))
        elif self.pos_class == 'all':
            empty[vehicles == 1] = 0
            empty[lane == 1] = 0
            empty[road == 1] = 0
            label = np.stack((vehicles, road, lane, empty))

        return label, ood[None], mapped_epistemic

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        agent_path, agent_id, index = self.data[idx]

        images, intrinsics, extrinsics = self.get_input_data(index, agent_path)
        labels, ood, mapped_epistemic = self.get_label(index, agent_path)

        if self.return_info:
            return images, intrinsics, extrinsics, labels, ood, {
                'agent_number': agent_id,
                'agent_path': agent_path,
                'index': index
            }

        if self.map_uncertainty:
            return images, intrinsics, extrinsics, labels, ood, mapped_epistemic
        else:
            return images, intrinsics, extrinsics, labels, ood

def compile_data(set, version, dataroot, pos_class, batch_size=8, num_workers=16, is_train=False, seed=0, yaw=-1, **kwargs):
    data = CarlaDataset(os.path.join(dataroot, set), is_train, pos_class, **kwargs)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if version == 'mini':
        g = torch.Generator()
        g.manual_seed(seed)

        sampler = torch.utils.data.RandomSampler(data, num_samples=256, generator=g)

        loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            sampler=sampler,
            pin_memory=True,
        )
    else:
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=False,
        )

    return loader
