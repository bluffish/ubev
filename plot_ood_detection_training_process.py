import matplotlib.pyplot as plt
import torch
import numpy as np
import yaml
import os
import re
import tqdm

from models.evidential import Evidential
from tools.metrics import get_iou, roc_pr, unc_iou
from eval import eval

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('-?([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def plot_ood_detection_results(
        pt_path,
        model_name,
        backbone_name="lss",
        config_path="./configs/eval_carla_lss_evidential.yaml",
        save_path="plots",
    ):
    torch.manual_seed(0)
    np.random.seed(0)

    model = Evidential([0], backbone=backbone_name)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    config['ood'] = True
    config['gpus'] = [6, 7]
    config['pos_class'] = 'vehicle'
    target_class_name = "Vehicle"

    tags = ["Background mIOU", f"{target_class_name} mIOU", "OOD AUROC", "OOD AUPR",
            f"({target_class_name} IOU+OOD PR)/2", "Total Loss", "UCE/UFocal Loss", "OOD Reg."]

    sets = {
        "val_aug": "Validation set w/ Augmented OOD",
        "ood": "Test set w/ True OOD",
        "train_aug": "Training set w/ Augmented OOD",
    }

    epochs = np.linspace(0, 19, 20)

    fig, axs = plt.subplots(6, 4, figsize=(30, 36))

    for si, s in enumerate(sets):
        axs[si * 2, 2].set_title(sets[s], fontsize=16)

        for i in range(len(tags)):
            ax = axs[si * 2 + i // 4, i % 4]
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(tags[i], fontsize=12)
            ax.set_xticks(epochs)
            ax.set_xlim(-.5, 19.5)

            ax.tick_params(axis='both', which='both', bottom=True, left=True, labelsize=10)

    fig.suptitle(model_name, fontsize=24)

    checkpoints = sorted_alphanumeric(os.listdir(pt_path))

    for si, s in enumerate(sets):
        set_name = s

        # if s == 'val' or s == 'train':
        #     config['pseudo'] = True
        # if s == 'ood':
        #     config['pseudo'] = False
        #     set_name = "val"

        data = {tag: [] for tag in tags}

        for checkpoint in checkpoints:
            if checkpoint.endswith(".pt"):
                config['pretrained'] = os.path.join(pt_path, checkpoint)

                pred, gt, ood, al, ep, raw = eval(config, set_name, "mini", "../data/carla")

                ious = get_iou(pred, gt)
                ep = ep.squeeze(1)
                ood = ood.bool()
                _, _, _, _, auroc, ap, _ = roc_pr(ep, ood)
                total, ood_reg = model.loss_ood(raw, gt, ood.long())
                uce = total - ood_reg

                data["Background mIOU"].append(ious[0])
                data[f"{target_class_name} mIOU"].append(ious[1])
                data["OOD AUROC"].append(auroc)
                data["OOD AUPR"].append(ap)
                data[f"({target_class_name} IOU+OOD PR)/2"].append((ious[1]+ap)/2)
                data["Total Loss"].append(total)
                data["UCE/UFocal Loss"].append(uce)
                data["OOD Reg."].append(ood_reg)

                #data["OOD mIOU"].append(unc_iou(ep, ood, thresh=.5))

        for i, k in enumerate(data.keys()):
            ax = axs[si * 2 + i // 4, i % 4]
            ax.plot(data[k], 'o-', label=model_name)

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"{model_name}-ValidationTestTrain-ood-Curves.png"))
    fig.savefig(os.path.join(save_path, f"{model_name}-ValidationTestTrain-ood-Curves.svg"), format="svg")

    plt.clf()


if __name__ == "__main__":

    models_folder = "outputs_bin/carla/grid_aug"
    for model_folder_name in tqdm.tqdm(os.listdir(models_folder)):
        pt_path = os.path.join(models_folder, model_folder_name)
        if not os.path.exists(os.path.join(pt_path, "19.pt")):
            continue
        model_name = model_folder_name
        plot_ood_detection_results(pt_path=pt_path, model_name=model_name, save_path="plots/LSS_CARLA_grid_aug")
