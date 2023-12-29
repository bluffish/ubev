import json
import math
import os

import torchvision
from torch.utils.data import Subset

from tools.geometry import *


class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train):
        self.is_train = is_train
        self.return_info = False

        self.data_path = data_path

        self.mode = 'train' if self.is_train else 'val'

        self.vehicles = len(os.listdir(os.path.join(self.data_path, 'agents')))
        self.ticks = len(os.listdir(os.path.join(self.data_path, 'agents/0/back_camera')))
        self.offset = 0

        self.to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )

        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )

    def get_input_data(self, index, agent_path):
        images = []
        intrinsics = []
        extrinsics = []

        with open(os.path.join(agent_path, 'sensors.json'), 'r') as f:
            sensors = json.load(f)

        for sensor_name, sensor_info in sensors['sensors'].items():
            if sensor_info["sensor_type"] == "sensor.camera.rgb" and sensor_name != "birds_view_camera":
                image = Image.open(os.path.join(agent_path + sensor_name, f'{index}.png'))

                intrinsic = torch.tensor(sensor_info["intrinsic"])
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
                intrinsics.append(intrinsic)
                extrinsics.append(torch.tensor(extrinsic))
                image.close()

        images, intrinsics, extrinsics = (torch.stack(images, dim=0),
                                          torch.stack(intrinsics, dim=0),
                                          torch.stack(extrinsics, dim=0))

        return images, intrinsics, extrinsics

    def get_label(self, index, agent_path):
        label_r = Image.open(os.path.join(agent_path + "bev_semantic", f'{index}.png'))
        label = np.array(label_r)
        label_r.close()

        empty = np.ones(self.bev_dimension[:2])

        road = mask(label, (128, 64, 128))
        lane = mask(label, (157, 234, 50))
        vehicles = mask(label, (0, 0, 142))

        if np.sum(vehicles) < 5:
            lane = mask(label, (50, 234, 157))
            vehicles = mask(label, (142, 0, 0))

        ood = mask(label, (0, 0, 0))
        bounding_boxes = find_bounding_boxes(ood)
        ood = draw_bounding_boxes(bounding_boxes)

        empty[vehicles == 1] = 0
        empty[road == 1] = 0
        empty[lane == 1] = 0
        label = np.stack((vehicles, road, lane, empty))

        return torch.tensor(label.copy()), torch.tensor(ood)

    def __len__(self):
        return self.ticks * self.vehicles

    def __getitem__(self, index):
        agent_number = math.floor(index / self.ticks)
        agent_path = os.path.join(self.data_path, f"agents/{agent_number}/")
        index = (index + self.offset) % self.ticks

        images, intrinsics, extrinsics = self.get_input_data(index, agent_path)
        labels, ood = self.get_label(index, agent_path)

        if self.return_info:
            return images, intrinsics, extrinsics, labels, ood, {
                'agent_number': agent_number,
                'agent_path': agent_path,
                'index': index
            }

        return images, intrinsics, extrinsics, labels, ood


def compile_data(version, dataroot, batch_size=8, num_workers=16, ood=False, pseudo=False):
    if pseudo:
        print("USING PSEUDO")
        train_data = CarlaDataset(os.path.join(dataroot, "train_aug"), True)
        val_data = CarlaDataset(os.path.join(dataroot, "val_aug"), False)
    elif ood:
        train_data = CarlaDataset(os.path.join(dataroot, "ood"), False)
        val_data = CarlaDataset(os.path.join(dataroot, "ood"), False)
    else:
        train_data = CarlaDataset(os.path.join(dataroot, "train"), True)
        val_data = CarlaDataset(os.path.join(dataroot, "val"), False)

    if version == 'mini':
        g = torch.Generator()
        g.manual_seed(0)

        train_sampler = torch.utils.data.RandomSampler(train_data, num_samples=128, generator=g)
        val_sampler = torch.utils.data.RandomSampler(val_data, num_samples=128, generator=g)

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            sampler=train_sampler
        )

        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            sampler=val_sampler,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

    return train_loader, val_loader
