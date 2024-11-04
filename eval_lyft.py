from eval import *
import seaborn as sns
import sys

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


def get(config):
    dataroot = f"../data/{config['dataset']}"
    preds, labels, oods, aleatoric, epistemic, raw = eval(config, "ood_test", "mini", dataroot, disable_tqdm=True)

    iou = get_iou(preds, labels, exclude=oods)[0]
    ece = expected_calibration_error(preds, labels, exclude=oods)[2]
    mis = get_mis(preds, labels)
    roc, pr, _, fpr95 = roc_pr(aleatoric, mis, exclude=oods)[4:8]
    ood_auroc, ood_aupr, _, ood_fpr95 = roc_pr(epistemic, oods)[4:8]
    return [iou, ece, roc, pr, fpr95, ood_auroc, ood_aupr, ood_fpr95]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False, type=int)
    parser.add_argument('-l', '--logdir', required=False, type=str)
    parser.add_argument('-b', '--batch_size', required=False, type=int)

    parser.add_argument('-p', '--pretrained', required=False, type=str)
    parser.add_argument('-e', '--ens', nargs='+', required=False, type=str)
    parser.add_argument('--ep_mode', required=False, type=str)

    parser.add_argument('-c', '--pos_class', default="vehicle", required=False, type=str)

    args = parser.parse_args()

    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')

    print(f"Using config {args.config}")
    config = get_config(args)

    config['dataset'] = "lyft"
    if 'ens' in config:
        config['ensemble'] = config['ens']
    elif 'pre' in config:
        config['pretrained'] = config['pre']
    config['alt'] = False

    print(config)

    out = get(config)

    ans = '=SPLIT("'

    for i in range(len(out)):
        ans += f"{out[i]:.3g},"

    ans = ans[0:len(ans)-1]+'", ",")'
    sys.stdout = save_stdout

    print(ans)