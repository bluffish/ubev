import torch.nn.functional
from tools.viz import *
from train import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(0)
np.random.seed(0)

torch.set_printoptions(precision=10)


def eval(config, set, split, dataroot):
    global colors, n_classes, classes, weights

    if config['binary']:
        colors = torch.tensor([
            [0, 0, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 0],
            [255, 255, 255],
        ])

        n_classes, classes = 2, ["vehicle", "background"]
        weights = torch.tensor([2, 1])
        change_params(n_classes, classes, colors, weights)

    train_loader, val_loader = datasets[config['dataset']](
        split, dataroot,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        ood=config['ood'],
        pseudo=config['pseudo'],
        binary=config['binary']
    )

    model = models[config['type']](
        config['gpus'],
        backbone=config['backbone'],
        n_classes=n_classes
    )

    if set == 'val':
        print("Using validation set")
        loader = val_loader
    elif set == 'train':
        print("Using train set")
        loader = train_loader
    else:
        raise NotImplementedError()

    model.load(torch.load(config['pretrained']))

    print("--------------------------------------------------")
    print(f"Running eval on {split}")
    print(f"Using GPUS: {config['gpus']}")
    print(f"Loader: {len(loader.dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Output directory: {config['logdir']} ")
    print(f"Pretrained: {config['pretrained']} ")
    print("--------------------------------------------------")

    os.makedirs(config['logdir'], exist_ok=True)

    predictions, ground_truths, oods, aleatoric, epistemic, raw = [], [], [], [], [], []

    with torch.no_grad():
        for images, intrinsics, extrinsics, labels, ood in tqdm(loader, desc="Running validation"):
            model.eval()
            model.training = False

            outs = model(images, intrinsics, extrinsics).detach().cpu()
            predictions.append(model.activate(outs))
            ground_truths.append(labels)
            oods.append(ood.bool())
            aleatoric.append(model.aleatoric(outs))
            epistemic.append(model.epistemic(outs))
            raw.append(outs)

            if config['ood']:
                save_unc(model.epistemic(outs)/model.epistemic(outs).max(), ood, config['logdir'])
            else:
                save_unc(model.aleatoric(outs), model.activate(outs).argmax(dim=1) != labels.argmax(dim=1),
                         config['logdir'])

            save_pred(model.activate(outs), labels, config['logdir'])

    return (torch.cat(predictions, dim=0),
            torch.cat(ground_truths, dim=0),
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
    parser.add_argument('-s', '--split', default="mini", required=False, type=str)
    parser.add_argument('-p', '--pretrained', required=False, type=str)
    parser.add_argument('-o', '--ood', default=False, action='store_true')
    parser.add_argument('-m', '--metric', default="rocpr", required=False)
    parser.add_argument('-r', '--save', default=False, action='store_true')
    parser.add_argument('--num_workers', required=False, type=int)
    parser.add_argument('--set', default="val", required=False, type=str)
    parser.add_argument('-t', '--tsne', default=False, action='store_true')
    parser.add_argument('--pseudo', default=False, action='store_true')
    parser.add_argument('--binary', default=False, action='store_true')

    args = parser.parse_args()

    print(f"Using config {args.config}")
    config = get_config(args)

    if config['backbone'] == 'cvt':
        torch.backends.cudnn.enabled = False

    split = args.split
    metric = args.metric
    set = args.set

    dataroot = f"../data/{config['dataset']}"
    name = f"{config['backbone']}_{config['type']}"

    predictions, ground_truth, oods, aleatoric, epistemic, raw = eval(config, set, split, dataroot)

    if args.save:
        torch.save(predictions, os.path.join(config['logdir'], 'prediction.pt'))
        torch.save(ground_truth, os.path.join(config['logdir'], 'ground_truth.pt'))
        torch.save(oods, os.path.join(config['logdir'], 'oods.pt'))
        torch.save(aleatoric, os.path.join(config['logdir'], 'aleatoric.pt'))
        torch.save(epistemic, os.path.join(config['logdir'], 'epistemic.pt'))
        torch.save(raw, os.path.join(config['logdir'], 'raw.pt'))

    iou = get_iou(predictions, ground_truth, exclude=oods)
    ece = ece(predictions, ground_truth, exclude=oods)
    brier = brier_score(predictions, ground_truth, exclude=oods)

    print(f"IOU: {iou}")
    print(f"Brier: {brier:.3f}")
    print(f"ECE: {ece:.3f}")

    if config['ood']:
        uncertainty_scores = epistemic.squeeze(1)
        uncertainty_labels = oods

        print("EVAL OOD")
    else:
        uncertainty_scores = aleatoric.squeeze(1)
        uncertainty_labels = torch.argmax(ground_truth, dim=1).cpu() != torch.argmax(predictions, dim=1).cpu()

    if metric == 'patch':
        pavpu, agc, ugi, thresholds, au_pavpu, au_agc, au_ugi = patch_metrics(uncertainty_scores, uncertainty_labels)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        ax1.plot(thresholds, agc, 'g.-', label=f"AU-p(accurate|certain): {au_agc:.3f}")
        ax1.set_xlabel('Uncertainty Threshold')
        ax1.set_ylabel('p(accurate|certain)')
        ax1.legend(frameon=True)
        ax1.set_ylim(-0.05, 1.05)

        ax2.plot(thresholds, ugi, 'r.-', label=f"AU-p(uncertain|inaccurate): {au_ugi:.3f}")
        ax2.set_xlabel('Uncertainty Threshold')
        ax2.set_ylabel('p(uncertain|inaccurate)')
        ax2.legend(frameon=True)
        ax2.set_ylim(-0.05, 1.05)

        ax3.plot(thresholds, pavpu, 'b.-', label=f"AU-PAvPU: {au_pavpu:.3f}")
        ax3.set_xlabel('Uncertainty Threshold')
        ax3.set_ylabel('PAVPU')
        ax3.legend(frameon=True)
        ax3.set_ylim(-0.05, 1.05)

        fig.suptitle(f"{'OOD' if config['ood'] else 'Misclassification'} - {name}")

        save_path = os.path.join(config['logdir'], f"patch_{'o' if config['ood'] else 'm'}_{name}")

        print(
            f"AU-PAvPU: {au_pavpu:.3f}, AU-p(accurate|certain): {au_agc:.3f}, AU-P(uncertain|inaccurate): {au_ugi:.3f}")
    elif metric == "rocpr":
        fpr, tpr, rec, pr, auroc, aupr, no_skill = roc_pr(uncertainty_scores, uncertainty_labels)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        plot_roc(axs[0], fpr, tpr, auroc)
        plot_pr(axs[1], rec, pr, aupr, no_skill)

        fig.suptitle(f"{'OOD' if config['ood'] else 'Misclassification'} - {name}")

        save_path = os.path.join(config['logdir'], f"rocpr_{'o' if config['ood'] else 'm'}_{name}")

        print(f"UNCERTAINTY IOU: {unc_iou(uncertainty_scores, uncertainty_labels, thresh=.5)}")
        print(f"AUROC: {auroc:.3f} AP: {aupr:.3f}")
    elif metric == "grid":
        fig, axs = plt.subplots(3, 3, figsize=(18, 18))

        fpr, tpr, rec, pr, auroc, aupr, no_skill = roc_pr(uncertainty_scores, uncertainty_labels)

        plot_roc(axs[0, 0], fpr, tpr, auroc)
        plot_pr(axs[1], rec, pr, aupr)

        axs[0, 2].hist(epistemic[~oods.unsqueeze(1).bool()].ravel().cpu().numpy(), label="ID", range=(0, 1), alpha=.7, bins=25, density=True, histtype='stepfilled')
        axs[0, 2].hist(epistemic[oods.unsqueeze(1).bool()].ravel().cpu().numpy(), label="OOD", range=(0, 1), alpha=.7, bins=25, density=True, histtype='stepfilled')
        axs[0, 2].legend()
        axs[0, 2].set_title("Epistemic Unc. Dist.")

        uncertainty_scores = aleatoric.squeeze(1)
        uncertainty_labels = torch.argmax(ground_truth, dim=1).cpu() != torch.argmax(predictions, dim=1).cpu()
        fpr, tpr, rec, pr, auroc, ap, no_skill = roc_pr(uncertainty_scores, uncertainty_labels)

        axs[1, 0].plot(fpr, tpr, 'r-', label=f'AUROC={auroc:.3f}')
        axs[1, 0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='No Skill - 0.500')
        axs[1, 0].set_xlabel('False Positive Rate')
        axs[1, 0].set_ylabel('True Positive Rate')
        axs[1, 0].tick_params(axis='x', which='both', bottom=True)
        axs[1, 0].tick_params(axis='y', which='both', left=True)
        axs[1, 0].legend()
        axs[1, 0].set_title("Misc. AUROC")

        axs[1, 1].step(rec, pr, 'r-', where='post', label=f'AP={ap:.3f}')
        axs[1, 1].plot([0, 1], [no_skill, no_skill], linestyle='--', color='gray', label=f'No Skill - {no_skill:.3f}')
        axs[1, 1].set_xlabel('Recall')
        axs[1, 1].set_ylabel('Precision')
        axs[1, 1].tick_params(axis='x', which='both', bottom=True)
        axs[1, 1].tick_params(axis='y', which='both', left=True)
        axs[1, 1].legend()
        axs[1, 1].set_title("Misc. AP")

        mis = (torch.argmax(ground_truth, dim=1).cpu() != torch.argmax(predictions, dim=1).cpu()).unsqueeze(1)
        axs[1, 2].hist(aleatoric[mis].ravel().cpu().numpy(), label="Misclassified", range=(0, 1), alpha=.7, bins=25, density=True, histtype='stepfilled',)
        axs[1, 2].hist(aleatoric[~mis].ravel().cpu().numpy(), label="Correctly Classified", range=(0, 1), alpha=.7, bins=25, density=True, histtype='stepfilled',)
        axs[1, 2].legend()
        axs[1, 2].set_title("Aleatoric Unc. Dist.")

        if config['binary']:
            classes = ["Vehicle", "Backg."]
            axs[2, 0].bar(classes, iou, color=['blue', 'red'])
        else:
            classes = ["Vehicle", "Road", "Lane", "Backg."]
            axs[2, 0].bar(classes, iou, color=['blue', 'green', 'red', 'purple'])

        axs[2, 0].set_title("Class IOUs")
        add_labels(classes, iou, axs[2, 0])

        plot_acc_calibration(axs[2, 1], ~oods.unsqueeze(1), predictions, ground_truth)

        # midpoints, accuracies, ece = calibration_curve(predictions.numpy(), ground_truth.numpy(), bins=10)

        # axs[2, 1].bar(midpoints, accuracies, width=1.0 / float(10), align='center', lw=1, ec='#000000', fc='#2233aa',
        #         alpha=1, label=f'ECE={ece:.5f}', zorder=0)
        # axs[2, 1].scatter(midpoints, accuracies, lw=2, ec='black', fc="#ffffff", zorder=2)
        # axs[2, 1].plot(np.linspace(0, 1.0, 20), np.linspace(0, 1.0, 20), '--', lw=2, alpha=.7, color='gray',
        #          label='Perfectly Calibrated', zorder=1)
        # axs[2, 1].set_xlim(0.0, 1.0)
        # axs[2, 1].set_ylim(0.0, 1.0)
        # axs[2, 1].set_xlabel('Confidence')
        # axs[2, 1].set_ylabel('Accuracy')
        # axs[2, 1].set_title("Calibration Plot")
        # axs[2, 1].set_xticks(midpoints, rotation=-45)
        # axs[2, 1].legend()

        axs[2, 2].axis("off")

        fig.tight_layout()
        save_path = os.path.join(config['logdir'], f"grid_{name}")
    else:
        raise ValueError("Please pick a valid metric.")

    fig.savefig(f"{save_path}.png", bbox_inches='tight')
    fig.savefig(f"{save_path}.pdf", bbox_inches='tight', format='pdf')

    print(f"Graph saved to {save_path}")
