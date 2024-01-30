import torch.nn.functional
from tools.viz import *
from train import *
import seaborn as sns

sns.set_style('white')
sns.set_palette('muted')
sns.set_context(
    "notebook",
    font_scale=1.25,
    rc={"lines.linewidth": 2.5}
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.multiprocessing.set_sharing_strategy('file_system')

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

torch.manual_seed(0)
np.random.seed(0)

torch.set_printoptions(precision=10)


def eval(config, set, split, dataroot):
    for gpu in config['gpus']:
        torch.inverse(torch.ones((1, 1), device=gpu))

    classes, n_classes, weights = change_params(config)

    loader = datasets[config['dataset']](
        set, split, dataroot, config['pos_class'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )

    print(f"Using set: {set}")

    model = models[config['type']](
        config['gpus'],
        backbone=config['backbone'],
        n_classes=n_classes
    )

    if config['type'] == 'ensemble':
        state_dicts = [torch.load(path) for path in config['ensemble']]
        model.load(state_dicts)
    else:
        model.load(torch.load(config['pretrained']))

    model.eval()

    print("--------------------------------------------------")
    print(f"Running eval on {split}")
    print(f"Using GPUS: {config['gpus']}")
    print(f"Loader: {len(loader.dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Output directory: {config['logdir']} ")
    print(f"Pretrained: {config['pretrained']} ")
    print("--------------------------------------------------")

    os.makedirs(config['logdir'], exist_ok=True)

    preds, labels, oods, aleatoric, epistemic, raw = [], [], [], [], [], []

    with torch.no_grad():
        for images, intrinsics, extrinsics, label, ood in tqdm(loader, desc="Running validation"):
            out = model(images, intrinsics, extrinsics).detach().cpu()
            pred = model.activate(out)

            preds.append(pred)
            labels.append(label)
            oods.append(ood.bool())
            aleatoric.append(model.aleatoric(out))
            epistemic.append(model.epistemic(out))
            raw.append(out)

            save_unc(model.epistemic(out), ood, config['logdir'], "epistemic.png", "ood.png")
            save_unc(model.aleatoric(out), get_mis(pred, label), config['logdir'], "aleatoric.png", "mis.png")
            save_pred(pred, label, config['logdir'])

    return (torch.cat(preds, dim=0),
            torch.cat(labels, dim=0),
            torch.cat(oods, dim=0),
            torch.cat(aleatoric, dim=0),
            torch.cat(epistemic, dim=0),
            torch.cat(raw, dim=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False, type=int)
    parser.add_argument('-l', '--logdir', required=False, type=str)
    parser.add_argument('-b', '--batch_size', required=False, type=int)
    parser.add_argument('-p', '--pretrained', required=False, type=str)
    parser.add_argument('-e', '--ensemble', nargs='+', required=False, type=str)
    parser.add_argument('--num_workers', required=False, type=int)
    parser.add_argument('--pseudo', default=False, action='store_true')
    parser.add_argument('-c', '--pos_class', default="vehicle", required=False, type=str)

    args = parser.parse_args()

    print(f"Using config {args.config}")
    config = get_config(args)

    dataroot = f"../data/{config['dataset']}"

    preds, labels, oods, aleatoric, epistemic, raw = eval(config, "ood_test", "mini", dataroot)

    mis = get_mis(preds, labels)
    iou = get_iou(preds, labels, exclude=oods)[0]
    ece = expected_calibration_error(preds, labels, exclude=oods)[2]
    roc, pr = roc_pr(aleatoric, mis, exclude=oods)[4:6]

    ood_m_auroc = 0
    ood_m_aupr = 0
    k = 0
    for i in range(epistemic.shape[0]):
        if oods[i].sum() == 0:
            continue
        (a, b) = roc_pr(epistemic[i].unsqueeze(0), oods[i].unsqueeze(0))[4:6]
        ood_m_auroc += a
        ood_m_aupr += b
        k += 1

    ood_auroc, ood_aupr = roc_pr(epistemic, oods)[4:6]
    _, _, oods_o, _, epistemic_o, _ = eval(config, "ood", "mini", dataroot)
    ood_auroc_o, ood_aupr_o = roc_pr(epistemic_o, oods_o)[4:6]

    print(round(iou, 3),
          round(ece.item(), 6),
          round(roc, 3),
          round(pr, 3),
          round(ood_m_auroc/k, 3),
          round(ood_m_aupr/k, 3),
          round(ood_auroc, 3),
          round(ood_aupr, 3),
          round(ood_auroc_o, 3),
          round(ood_aupr_o, 3))

