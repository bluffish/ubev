import json
import math
import os
import random

import torchvision
from torch.utils.data import Subset

from tools.geometry import *


def assign_weather_indices(total_length, weather_list, switch_frequency, repeat_times):
    """
    Assigns indices for each weather type with the first section of the first weather being (n-1) ticks long,
    all others being n ticks long, and repeats the sequence `repeat_times` times.

    Args:
        total_length (int): Total length of the dataset (number of frames per cycle).
        weather_list (list): List of weather types.
        switch_frequency (int): Number of frames for each weather type (n ticks).
        repeat_times (int): Number of times to repeat the sequence.

    Returns:
        dict: A dictionary with weather types as keys and lists of indices as values.
    """
    # Initialize the dictionary to store indices for each weather type
    weather_indices = {weather: [] for weather in weather_list}

    # Calculate indices for a single cycle
    single_cycle_indices = {weather: [] for weather in weather_list}
    current_index = 0
    first_weather_adjustment = switch_frequency - 1  # First section is (n-1) ticks
    is_first_weather = True

    while current_index < total_length:
        for weather in weather_list:
            # Determine the length of the current section
            if is_first_weather:
                section_length = first_weather_adjustment
                is_first_weather = False
            else:
                section_length = switch_frequency

            # Add indices for the current weather
            section_end = min(current_index + section_length, total_length)
            single_cycle_indices[weather].extend(range(current_index, section_end))

            # Update the current index
            current_index = section_end

            # Break if we've reached the total length
            if current_index >= total_length:
                break

    # Repeat the indices for each cycle
    total_length_per_cycle = sum(len(indices) for indices in single_cycle_indices.values())
    for cycle in range(repeat_times):
        for weather, indices in single_cycle_indices.items():
            offset = cycle * total_length_per_cycle
            weather_indices[weather].extend(idx + offset for idx in indices)

    return weather_indices


def assign_town_indices(total_length, town_list, switch_frequency):
    # Initialize the dictionary to store indices for each weather type
    town_indices = {town: [] for town in town_list}

    # Assign indices to each weather type
    for idx in range(total_length):
        # Determine the current weather type based on the frame index
        current_town = town_list[(idx // switch_frequency) % len(town_list)]
        town_indices[current_town].append(idx)

    return town_indices


class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train, pos_class, weather=None, town=None):
        self.is_train = is_train
        self.return_info = False
        self.pos_class = pos_class

        self.data_path = data_path

        self.mode = 'train' if self.is_train else 'val'

        self.vehicles = len(os.listdir(os.path.join(self.data_path, 'agents')))
        # self.vehicles = 1
        self.ticks = len(os.listdir(os.path.join(self.data_path, 'agents/0/back_camera')))
        self.offset = 0

        self.to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )

        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution, bev_start_position, bev_dimension
        )

        self.towns = [
            "Town10HD_Opt",
            #"Town03_Opt",
            "Town05_Opt",
            "Town07_Opt",
            "Town02_Opt"
        ]

        self.weathers = [
            "ClearNoon",
            "CloudyNoon",
            "WetNoon",
            "MidRainyNoon",
            "SoftRainNoon",
            "ClearSunset",
            "CloudySunset",
            "WetSunset",
            "WetCloudySunset",
            "SoftRainSunset",
        ]

        self.weather_idxs = assign_weather_indices(self.ticks, self.weathers, 2, self.vehicles)
        self.town_idxs = assign_town_indices(self.ticks * self.vehicles, self.towns, 20)

        self.weather = weather
        self.town = town

        # print(self.weather_idxs)

    def get_input_data(self, index, agent_path):
        images = []
        intrinsics = []
        extrinsics = []

        with open(os.path.join(agent_path, 'sensors.json'), 'r') as f:
            sensors = json.load(f)

        for sensor_name, sensor_info in sensors['sensors'].items():
            if sensor_info["sensor_type"] == "sensor.camera.rgb" and sensor_name != "birds_view_camera":
                image = Image.open(os.path.join(agent_path + sensor_name, f'{index}.png'))
                image.save(f"cam/{sensor_name}_{index}.png")

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

        ood = mask(label, (0, 0, 0)) | mask(label, (50, 100, 144))
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

        return label, ood[None]

    def __len__(self):
        if self.weather is not None and self.town is not None:
            raise Exception

        if self.weather is not None:
            return len(self.weather_idxs[self.weather])
        if self.town is not None:
            return len(self.town_idxs[self.town])
        else:
            return self.ticks * self.vehicles

    def __getitem__(self, index):
        if self.weather is not None:
            index = self.weather_idxs[self.weather][index]
        elif self.town is not None:
            index = self.town_idxs[self.town][index]

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


def compile_data(set, version, dataroot, pos_class, batch_size=8, num_workers=16, is_train=False, seed=0, yaw=-1, nuscenes_c=None, true_ood=None, alt=False, weather=None, town=None):
    data = CarlaDataset(os.path.join(dataroot, set), is_train, pos_class, weather=weather, town=town)
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
            shuffle=False,
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
