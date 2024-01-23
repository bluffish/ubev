from eval import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False, type=int)
    parser.add_argument('-l', '--logdir', required=False, type=str)
    parser.add_argument('-b', '--batch_size', required=False, type=int)
    parser.add_argument( '--split', default="mini", required=False, type=str)
    parser.add_argument('-p', '--pretrained', required=False, type=str)
    parser.add_argument('-m', '--metric', default="rocpr", required=False)
    parser.add_argument('-r', '--save', default=False, action='store_true')
    parser.add_argument('--num_workers', required=False, type=int)
    parser.add_argument('--pseudo', default=False, action='store_true')
    parser.add_argument('-c', '--pos_class', default="vehicle", required=False, type=str)
    parser.add_argument( '-n', '--name', default="model", required=False, type=str)
    parser.add_argument('-o', '--ood', default=False, action='store_true')
    args = parser.parse_args()

    print(f"Using config {args.config}")
    config = get_config(args)

    if config['backbone'] == 'cvt':
        torch.backends.cudnn.enabled = False
    else:
        torch.backends.cudnn.enabled = True

    dataroot = f"../data/{config['dataset']}"
    split, metric = args.split, args.metric

    if args.ood:
        sets = ["ood", "val_aug_stable", "train_aug_stable"]
        set_names = ["OOD", "Val Aug", "Train Aug"]

        fig, axs = plt.subplots(3, 2, figsize=(3*6, 3*6))
        fig.suptitle(args.name)

        for i, set in enumerate(sets):
            preds, labels, oods, aleatoric, epistemic, raw = eval(config, set, split, dataroot)

            plot_roc_pr(epistemic, oods, axs=(axs[i, 0], axs[i, 1]))
            axs[i, 0].set_title(f"{set_names[i]} ROC Curve")
            axs[i, 1].set_title(f"{set_names[i]} PR Curve")

        fig.tight_layout()
        fig.savefig(os.path.join(config['logdir'], f"{args.name}_ood_metrics.png"))
        fig.savefig(os.path.join(config['logdir'], f"{args.name}_ood_metrics.pdf"), format='pdf')
    else:
        sets = ["test", "val", "train"]
        set_names = ["Test", "Val", "Train"]

        fig, axs = plt.subplots(9, 3, figsize=(3*6, 9*6))
        fig.suptitle(args.name)

        for i, set in enumerate(sets):
            preds, labels, oods, aleatoric, epistemic, raw = eval(config, set, split, dataroot)

            mis = get_mis(preds, labels)
            plot_roc_pr(aleatoric, mis, exclude=oods, axs=(axs[i*3, 0], axs[i*3, 1]))
            axs[i*3, 0].set_title(f"{set_names[i]} ROC Curve")
            axs[i*3, 1].set_title(f"{set_names[i]} PR Curve")
            axs[i*3, 2].set_title(f"{set_names[i]} Calibration Curve")

            plot_ece(preds, labels, ax=axs[i*3, 2], exclude=oods)
            plot_patch(aleatoric, mis, axs=axs[i*3+1, :])
            plot_patch(aleatoric, mis, axs=axs[i*3+2, :], quantile=True)

            axs[i*3+1, 0].set_title(f"{set_names[i]} p(accurate|certain) with thresholds")
            axs[i*3+1, 1].set_title(f"{set_names[i]} p(uncertain|inaccurate) with thresholds")
            axs[i*3+1, 2].set_title(f"{set_names[i]} PAvPU with thresholds")
            axs[i*3+2, 0].set_title(f"{set_names[i]} p(accurate|certain) with quantiles")
            axs[i*3+2, 1].set_title(f"{set_names[i]} p(uncertain|inaccurate) with quantiles")
            axs[i*3+2, 2].set_title(f"{set_names[i]} PAvPU with quantiles")

        fig.tight_layout()
        fig.savefig(os.path.join(config['logdir'], f"{args.name}_mis_metrics.png"))
        fig.savefig(os.path.join(config['logdir'], f"{args.name}_mis_metrics.pdf    "), format='pdf')