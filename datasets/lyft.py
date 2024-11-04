from lyft_dataset_sdk.lyftdataset import LyftDataset
from datasets.nuscenes import *


def get_lyft(version, dataroot):
    # dataroot = os.path.join(dataroot, version)
    lyft = LyftDataset(data_path=dataroot, json_path=os.path.join(dataroot, 'train_data'), verbose=False)

    return lyft, dataroot


def compile_data(set, version, dataroot, pos_class, batch_size=8, num_workers=16, seed=0, yaw=180, is_train=False, true_ood=None, alt=False):
    if set == "train":
        ind, ood, pseudo, is_train = True, False, False, True
    elif set == "val":
        ind, ood, pseudo, is_train = True, False, False, False
    elif set == "train_aug":
        ind, ood, pseudo, is_train = False, False, True, True
    elif set == "val_aug":
        ind, ood, pseudo, is_train = False, False, True, False
    elif set == "train_comb":
        ind, ood, pseudo, is_train = True, False, True, True
    elif set == "val_comb":
        ind, ood, pseudo, is_train = True, False, True, False
    elif set == "train_full":
        ind, ood, pseudo, is_train = True, True, True, True
    elif set == "val_full":
        ind, ood, pseudo, is_train = True, True, True, False
    elif set == "ood":
        ind, ood, pseudo, is_train = False, True, False, False
    elif set == "test":
        ind, ood, pseudo, is_train = True, False, False, False
    elif set == "ood_test":
        ind, ood, pseudo, is_train = True, True, False, False
    else:
        raise NotImplementedError(f"Dataset {set} not exist.")

    lyft, dataroot = get_lyft('trainval', dataroot)

    data = NuScenesDataset(lyft, is_train, pos_class, ind=ind, ood=ood, pseudo=pseudo, yaw=yaw, true_ood=true_ood, alt=alt)
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
            pin_memory=True
        )

    return loader
