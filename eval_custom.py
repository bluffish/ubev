from eval import *
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False, type=int)
    parser.add_argument('-l', '--logdir', required=False, type=str)
    parser.add_argument('-b', '--batch_size', required=False, type=int)
    parser.add_argument('-p', '--pretrained', required=False, type=str)
    parser.add_argument('-e', '--ensemble', nargs='+', required=False, type=str)
    parser.add_argument('--pseudo', default=False, action='store_true')
    parser.add_argument('-c', '--pos_class', default="vehicle", required=False, type=str)

    args = parser.parse_args()

    print(f"Using config {args.config}")
    config = get_config(args)

    dataroot = f"../data/{config['dataset']}"

    preds, labels, oods, aleatoric, epistemic, raw = eval(config, "ood_test", "mini", dataroot)

    if config['pos_class'] == 'vehicle':
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

        print(f'&{iou:.3g}&{ece:.3g}&{roc:.3g}&{pr:.3g}&{ood_auroc:.3g}&{ood_aupr:.3g}\\\\')
        print(f'=SPLIT("{iou:.3g},{ece:.3g},{roc:.3g},{pr:.3g},{ood_m_auroc/k:.3g},{ood_m_aupr/k:.3g},{ood_auroc:.3g},{ood_aupr:.3g},{ood_auroc_o:.3g},{ood_aupr_o:.3g}", ",")')
    elif config['pos_class'] == 'road':

        mis = get_mis(preds, labels)
        iou = get_iou(preds, labels, exclude=oods)[0]
        ece = expected_calibration_error(preds, labels, exclude=oods)[2]
        roc, pr = roc_pr(aleatoric, mis, exclude=oods)[4:6]
        print(f'=SPLIT("{iou:.3g},{ece:.3g},{roc:.3g},{pr:.3g}", ",")')
