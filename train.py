import argparse
from time import time
import json

from tensorboardX import SummaryWriter
from tools.metrics import *
from tools.utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.multiprocessing.set_sharing_strategy('file_system')

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

print(torch.__version__)


def train(config, resume=False):
    classes, n_classes, weights = change_params(config)

    if config['loss'] == 'focal':
        config['learning_rate'] *= 4

    if config['ood']:
        if config['stable']:
            train_set = "train_aug_stable"
            val_set = "val_aug_stable"
        else:
            train_set = "train_comb"
            val_set = "val_comb"
    else:
        train_set = "train"
        val_set = "val"

    if 'train_set' in config:
        train_set = config['train_set']
    if 'val_set' in config:
        val_set = config['val_set']

    if config['backbone'] == 'lss':
        yaw = 0
    elif config['backbone'] == 'cvt':
        yaw = 180

    train_loader = datasets[config['dataset']](
        config['train_set'], split, dataroot, config['pos_class'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        is_train=True,
        seed=config['seed'],
        yaw=yaw
    )

    val_loader = datasets[config['dataset']](
        config['val_set'], split, dataroot, config['pos_class'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        is_train=False,
        seed=config['seed'],
        yaw=yaw
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

    starting_epoch = 0

    if config['mixed']:
        model.scaler = torch.cuda.amp.GradScaler(enabled=True)
        print("Using mixed precision")
    else:
        print("Using full precision")
    
    if 'pretrained' in config:
        model_load_path = config['pretrained']
    if resume:
        latest_pt_filename = find_latest_model_pt_filename(config['logdir'])
        if latest_pt_filename is None:
            print("!!! model checkpoints not found, cannot resume !!!")
        model_load_path = os.path.join(config['logdir'], latest_pt_filename)
        starting_epoch = int(os.path.basename(model_load_path).split('.')[0]) + 1

    if 'pretrained' in config:
        model.load(torch.load(model_load_path))
        print(f"Loaded pretrained weights: {model_load_path}")

    steps_per_epoch=len(train_loader.dataset) // config['batch_size']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        model.opt,
        div_factor=10,
        pct_start=.3,
        final_div_factor=10,
        max_lr=config['learning_rate'],
        epochs=config['num_epochs'],
        steps_per_epoch=steps_per_epoch
    )

    if 'pretrained' in config:
        pt_name = os.path.basename(model_load_path)
        scheduler_path = os.path.join(os.path.dirname(model_load_path), 'scheduler_'+pt_name)
        if os.path.isfile(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path))

    if 'gamma' in config:
        model.gamma = config['gamma']
        print(f"GAMMA: {model.gamma}")

    if 'ol' in config:
        model.ood_lambda = config['ol']
        print(f"OOD LAMBDA: {model.ood_lambda}")

    if 'k' in config:
        model.k = config['k']

    if 'beta' in config:
        model.beta_lambda = config['beta']
        print(f"Beta lambda is {model.beta_lambda}")

    if config['fast']:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        print("Using torch.compile")

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

    with open((os.path.join(config['logdir'], f'config.json')), 'w') as f:
        json.dump(config, f, indent=4)

    # enable to catch errors in loss function
    # torch.autograd.set_detect_anomaly(True)
    step = steps_per_epoch * starting_epoch
    for epoch in range(starting_epoch, config['num_epochs']):
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
                print(f"[{epoch}] {step} {loss.item()} {time()-t_0}")

                writer.add_scalar('train/step_time', time() - t_0, step)
                writer.add_scalar('train/loss', loss, step)

                if ood_loss is not None:
                    writer.add_scalar('train/ood_loss', ood_loss, step)

                if config['ood']:
                    save_unc(model.epistemic(outs) / model.epistemic(outs).max(), ood, config['logdir'], "epistemic.png", "ood.png")
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
            writer.add_scalar(f'val/{classes[i]}_iou', iou[i], epoch)

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
            writer.add_scalar(f"val/uce_loss", val_loss - ood_loss, epoch)

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
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(config['logdir'], f'scheduler_{epoch}.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('backbone', choices=['lss', 'cvt', 'fiery'], type=str)
    parser.add_argument('type', choices=['baseline', 'evidential', 'dropout', 'postnet', 'ensemble', 'energy'], type=str)
    parser.add_argument('dataset', choices=['nuscenes', 'carla'], type=str)

    parser.add_argument('-g', '--gpus', nargs='+', required=False, type=int)
    parser.add_argument('-l', '--logdir', default=None, required=False, type=str)
    parser.add_argument('-b', '--batch_size', default=32, required=False, type=int)
    parser.add_argument('-s', '--split', default="trainval", required=False, type=str)
    parser.add_argument('--num_workers', default=32, required=False, type=int)
    parser.add_argument('-s', '--split', default='trainval', required=False, type=str)
    parser.add_argument('-p', '--pretrained', required=False, type=str, help='Load pretrained model')
    parser.add_argument('-o', '--ood', default=False, action='store_true')
    parser.add_argument('-e', '--num_epochs', default=100, required=False, type=int)
    parser.add_argument('-c', '--pos_class', default='vehicle', required=False, type=str)
    parser.add_argument('-f', '--fast', default=False, action='store_true', help='Use torch.compile to speedup')
    parser.add_argument('--mixed', default=False, action='store_true', help='Use mixed percision')

    parser.add_argument('--seed', default=0, required=False, type=int)
    # Removed. Directly pass stable dataset name train_aug_stable and val_aug_stable with --train_set and --val_set instead.
    #parser.add_argument('--stable', default=False, action='store_true')

    parser.add_argument('--weight_decay', default=0.0000001, required=False, type=float)
    parser.add_argument('--learning_rate', default=0.01, required=False, type=float)

    parser.add_argument('--loss', default='ce', choices=['ce', 'focal', 'al'], required=False, type=str)
    parser.add_argument('--gamma', required=False, type=float)  # 0.1
    parser.add_argument('--beta', required=False, type=float, help='Evidential beta')   # 0.0001
    parser.add_argument('--ol', required=False, type=float, help='Evidential lambda')   # 0.1
    parser.add_argument('--k', required=False, type=float, help='Evidential k')         # 64

    parser.add_argument('--train_set', required=False, type=str)
    parser.add_argument('--val_set', required=False, type=str)

    parser.add_argument('--resume', default=False, action='store_true', help='Resume training from existing logdir. Do not use with --pretrained.')

    args = parser.parse_args()

    config = {}

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    config['mixed'] = False

    if config['backbone'] == 'cvt':
        torch.backends.cudnn.enabled = False

    for gpu in config['gpus']:
        torch.inverse(torch.ones((1, 1), device=gpu))

    split = args.split
    dataroot = f"../data/{config['dataset']}"

    if 'logdir' not in config or config['logdir'] is None:
        logdir = f"outputs_th/{config['dataset']}/{config['pos_class']}/{config['backbone']}_{config['type']}/{config['loss']}"
        if 'gamma' in config:
            logdir += f"_gamma={config['gamma']}"
        if 'beta' in config:
            logdir += f"_beta={config['beta']}"
        if 'ol' in config:
            logdir += f"_ol={config['ol']}"
        if 'k' in config:
            logdir += f"_k={config['k']}"
        config['logdir'] = logdir

    train(config, resume=args.resume)
