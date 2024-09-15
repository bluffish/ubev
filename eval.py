import torch.nn.functional
from tools.viz import *
from tools.utils import *
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


def eval(config, set, split, dataroot, disable_tqdm=False):
    for gpu in config['gpus']:
        torch.inverse(torch.ones((1, 1), device=gpu))
    if config['backbone'] == 'cvt':
        torch.backends.cudnn.enabled = False

    if config['backbone'] == 'lss' or config['backbone'] == 'pointbev':
        yaw = 0
    elif config['backbone'] == 'cvt':
        yaw = 180

    classes, n_classes, weights = change_params(config)

    true_ood = None
    if 'true_ood' in config:
        true_ood = config['true_ood']

    loader = datasets[config['dataset']](
        set, split, dataroot, config['pos_class'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        yaw=yaw,
        true_ood=true_ood
    )

    print(f"Using set: {set}")

    model = models[config['type']](
        config['gpus'],
        backbone=config['backbone'],
        n_classes=n_classes
    )

    if config['type'] == 'ensemble':
        state_dicts = [torch.load(path, map_location='cuda:0') for path in config['ensemble']]
        model.load(state_dicts)
    else:
        model.load(torch.load(config['pretrained'], map_location='cuda:0'))

    model.eval()

    if 'ep_mode' in config:
        model.ep_mode = config['ep_mode']

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

    total_ood = 0
    total_id = 0
    total = 0

    i = 0
    with torch.no_grad():
        for images, intrinsics, extrinsics, label, ood in tqdm(loader, desc="Running validation", disable=disable_tqdm):
            out = model(images, intrinsics, extrinsics).detach().cpu()

            total_ood += ood.sum()
            total_id += label[:, 0].sum()
            total += images.shape[0] * 200 * 200

            pred = model.activate(out)

            preds.append(pred)
            labels.append(label)
            oods.append(ood.bool())
            aleatoric.append(model.aleatoric(out))
            epistemic.append(model.epistemic(out)**2)
            raw.append(out)

            save_unc(model.epistemic(out), ood, config['logdir'], f"{i}_epistemic.png", f"{i}_ood.png")
            i += 1
            save_unc(model.aleatoric(out), get_mis(pred, label), config['logdir'], "aleatoric.png", "mis.png")
            save_pred(pred, label, config['logdir'])

    print("OOD Fraction", total_ood / total)
    print("IDT Fraction", total_id / total)

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
    parser.add_argument( '--split', default="mini", required=False, type=str)
    parser.add_argument('-s', '--set', default="ood", required=False, type=str)
    parser.add_argument('-p', '--pretrained', required=False, type=str)
    parser.add_argument('-e', '--ensemble', nargs='+', required=False, type=str)
    parser.add_argument('-m', '--metric', default="rocpr", required=False)
    parser.add_argument('-r', '--save', default=False, action='store_true')
    parser.add_argument('--num_workers', required=False, type=int)
    parser.add_argument('--ep_mode', required=False, type=str)
    parser.add_argument('--true_ood', nargs='+', required=False, type=str)

    parser.add_argument('--pseudo', default=False, action='store_true')
    parser.add_argument('-c', '--pos_class', default="vehicle", required=False, type=str)

    args = parser.parse_args()

    print(f"Using config {args.config}")
    config = get_config(args)

    dataroot = f"../data/{config['dataset']}"
    split, metric, set = args.split, args.metric, args.set

    preds, labels, oods, aleatoric, epistemic, raw = eval(config, set, split, dataroot)

    if args.save:
        torch.save(preds, os.path.join(config['logdir'], 'preds.pt'))
        torch.save(labels, os.path.join(config['logdir'], 'labels.pt'))
        torch.save(oods, os.path.join(config['logdir'], 'oods.pt'))
        torch.save(aleatoric, os.path.join(config['logdir'], 'aleatoric.pt'))
        torch.save(epistemic, os.path.join(config['logdir'], 'epistemic.pt'))
        torch.save(raw, os.path.join(config['logdir'], 'raw.pt'))

    iou = get_iou(preds, labels, exclude=oods)
    brier = brier_score(preds, labels, exclude=oods)

    mis = preds.argmax(dim=1) != labels.argmax(dim=1)
    mis_graph, mis_auroc, mis_aupr, mis_fpr95 = plot_roc_pr(aleatoric, mis, title="Misc ROC & PR", exclude=oods)
    ood_graph, ood_auroc, ood_aupr, ood_fpr95 = plot_roc_pr(epistemic, oods, title="OOD ROC & PR")
    ece_graph, ece = plot_ece(preds, labels, exclude=oods)

    # accs = accuracy_rejection_curve(preds, labels, 1-aleatoric[:, 0])
    # print(accs)

    print(f"IOU: {iou}, Brier: {brier:.5f}, ECE: {ece:.5f}")
    print(f"Mis AUROC={mis_auroc:.5f}, AUPR={mis_aupr:.5f}, FPR95={mis_fpr95:.5f}")
    print(f"OOD AUROC={ood_auroc:.5f}, AUPR={ood_aupr:.5f}, FPR95={ood_fpr95:.5f}")

    mis_graph.savefig(os.path.join(config['logdir'], "roc_pr_mis.png"))
    ood_graph.savefig(os.path.join(config['logdir'], "roc_pr_ood.png"))
    mis_graph.savefig(os.path.join(config['logdir'], "roc_pr_mis.pdf"), format="pdf")
    ood_graph.savefig(os.path.join(config['logdir'], "roc_pr_ood.pdf"), format="pdf")

    ece_graph.savefig(os.path.join(config['logdir'], "ece_calib.png"))
    ece_graph.savefig(os.path.join(config['logdir'], "ece_calib.pdf"), format="pdf")

    print(f"Graphs saved to {config['logdir']}")

    mds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    eces = []
    for m in mds:
        ece = expected_calibration_error(preds, labels, min_dist=m)[2].item()
        eces.append(ece)
    print(eces)
