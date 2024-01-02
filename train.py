import argparse
from time import time

from tensorboardX import SummaryWriter
from tools.metrics import *
from tools.utils import *
import importlib

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(0)
np.random.seed(0)


def train():
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

    if config['loss'] == 'focal':
        config['learning_rate'] *= 4

    train_loader, val_loader = datasets[config['dataset']](
        split, dataroot,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pseudo=config['ood'],
        ood=config['ood'],
        binary=config['binary']
    )

    model = models[config['type']](
        config['gpus'],
        backbone=config['backbone'],
        n_classes=n_classes,
        loss_type=config['loss'],
        weights=weights
    )

    model.opt = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    print("Using scheduler")

    if 'pretrained' in config:
        model.load(torch.load(config['pretrained']))
        print(f"Loaded pretrained weights: {config['pretrained']}")
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            model.opt,
            div_factor=10,
            pct_start=.3,
            final_div_factor=10,
            max_lr=config['learning_rate'],
            epochs=config['num_epochs'],
            steps_per_epoch=len(train_loader.dataset) // config['batch_size']
        )

    if 'gamma' in config:
        model.gamma = config['gamma']
        print(f"GAMMA: {model.gamma}")

    if 'ol' in config:
        model.ood_lambda = config['ol']
        print(f"OOD LAMBDA: {model.ood_lambda}")

    if 'k' in config:
        model.k = config['k']
        print(f"Scaling with {model.scale}")

    if 'scale' in config:
        model.scale = config['scale']
        print(f"Scaling with {model.scale} @ k={model.k}")

    print("--------------------------------------------------")
    print(f"Using GPUS: {config['gpus']}")
    print(f"Train loader: {len(train_loader.dataset)}")
    print(f"Val loader: {len(val_loader.dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Output directory: {config['logdir']} ")
    print(f"Using loss {config['loss']}")
    print("--------------------------------------------------")

    writer = SummaryWriter(logdir=config['logdir'])

    writer.add_text("config", str(config))

    step = 0
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(config['num_epochs']):
        model.train()

        writer.add_scalar('train/epoch', epoch, step)

        for images, intrinsics, extrinsics, labels, ood in train_loader:
            t_0 = time()
            ood_loss = None

            if config['ood']:
                outs, preds, loss, ood_loss = model.train_step_ood(images, intrinsics, extrinsics, labels, ood)
            else:
                outs, preds, loss = model.train_step(images, intrinsics, extrinsics, labels)

            step += 1

            if scheduler is not None:
                scheduler.step()

            if step % 10 == 0:
                print(f"[{epoch}] {step}", loss.item())

                writer.add_scalar('train/step_time', time() - t_0, step)
                writer.add_scalar('train/loss', loss, step)

                if ood_loss is not None:
                    writer.add_scalar('train/ood_loss', ood_loss, step)

                if config['ood']:
                    save_unc(model.epistemic(outs), ood, config['logdir'])
                save_pred(preds, labels, config['logdir'])

            if step % 50 == 0:
                iou = get_iou(preds.cpu(), labels)

                print(f"[{epoch}] {step}", "IOU: ", iou)

                for i in range(0, n_classes):
                    writer.add_scalar(f'train/{classes[i]}_iou', iou[i], step)

        model.eval()

        predictions, ground_truth, oods, aleatoric, epistemic, raw = run_loader(model, val_loader)
        iou = get_iou(predictions, ground_truth)

        for i in range(0, n_classes):
            writer.add_scalar(f'val/{classes[i]}_iou', iou[i], epoch)\

        print(f"Validation mIOU: {iou}")

        ood_loss = None
        if config['ood']:
            n_samples = len(raw)
            val_loss = 0
            ood_loss = 0

            for i in range(0, n_samples, config['batch_size']):
                raw_batch = raw[i:i + config['batch_size']].to(model.device)
                ground_truth_batch = ground_truth[i:i + config['batch_size']].to(model.device)
                oods_batch = oods[i:i + config['batch_size']].to(model.device)

                vl, ol = model.loss_ood(raw_batch, ground_truth_batch, oods_batch)

                val_loss += vl
                ood_loss += ol

            val_loss /= (n_samples / config['batch_size'])
            ood_loss /= (n_samples / config['batch_size'])

            writer.add_scalar('val/ood_loss', ood_loss, epoch)
            writer.add_scalar(f"val/loss", val_loss, epoch)
            writer.add_scalar(f"val/uce_loss", val_loss-ood_loss, epoch)

            uncertainty_scores = epistemic[:256].squeeze(1)
            uncertainty_labels = oods[:256].bool()

            fpr, tpr, rec, pr, auroc, aupr, _ = roc_pr(uncertainty_scores, uncertainty_labels)
            writer.add_scalar(f"val/ood_auroc", auroc, epoch)
            writer.add_scalar(f"val/ood_aupr", aupr, epoch)

            print(f"Validation OOD: AUPR={aupr}, AUROC={auroc}")
        else:
            n_samples = len(raw)
            val_loss = 0

            for i in range(0, n_samples, config['batch_size']):
                raw_batch = raw[i:i + config['batch_size']].to(model.device)
                ground_truth_batch = ground_truth[i:i + config['batch_size']].to(model.device)

                vl = model.loss(raw_batch, ground_truth_batch)

                val_loss += vl

            val_loss /= (n_samples / config['batch_size'])

            writer.add_scalar(f"val/loss", val_loss, epoch)

        if ood_loss is not None:
            print(f"Validation loss: {val_loss}, OOD Reg.: {ood_loss}")
        else:
            print(f"Validation loss: {val_loss}")

        model.save(os.path.join(config['logdir'], f'{epoch}.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False, type=int)
    parser.add_argument('-l', '--logdir', required=False, type=str)
    parser.add_argument('-b', '--batch_size', required=False, type=int)
    parser.add_argument('-s', '--split', default="trainval", required=False, type=str)
    parser.add_argument('-p', '--pretrained', required=False, type=str)
    parser.add_argument('-o', '--ood', default=False, action='store_true')
    parser.add_argument('-e', '--num_epochs', required=False, type=int)
    parser.add_argument('--loss', default="ce", required=False, type=str)
    parser.add_argument('--gamma', required=False, type=float)
    parser.add_argument('--ol', required=False, type=float)
    parser.add_argument('--scale', required=False, type=str)
    parser.add_argument('--k', required=False, type=float)
    parser.add_argument('--binary', default=False, action='store_true')

    args = parser.parse_args()

    print(f"Using config {args.config}")
    config = get_config(args)

    if config['backbone'] == 'cvt':
        torch.backends.cudnn.enabled = False
    else:
        torch.backends.cudnn.enabled = True

    split = args.split
    dataroot = f"../data/{config['dataset']}"

    train()
