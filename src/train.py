import argparse
import json
import os

import clip
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import autocast as autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from pathlib import Path
from dataset import ImageNetDataset
from utils import targetpad_transform, RunningAverage
from i2t_model import IMG2TEXT, Phi

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument('--optimizer', default='adam')
parser.add_argument('--clip_model_name', type=str, default='ViT-L/14')  # "ViT-L/14"
parser.add_argument('--pre_dataset', type=str, default="ImageNetDataset")  # ImageNetDataset
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--ImageNetPath', type=str, default="/data/ImageNet/")
parser.add_argument('--imgCaptionPath', type=str, default="./blip_pairs.json")

parser.add_argument('--model_dir', default='./save_model',
                    help="Directory containing params.json")
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--preprocess_type', type=str, default="targetpad")

parser.add_argument("--validation_frequency", default=1, type=int, help="Validation frequency expressed in epochs")
parser.add_argument("--save_training", default=True, type=bool, help="Whether save the model checkpoints or not")
parser.add_argument("--save_frequency", default=1, type=int, help="Saving frequency expressed in epochs")

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epochs', type=int, default=80)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--tf_layer', type=int, default=3)
parser.add_argument('--tf_head', type=int, default=1)
parser.add_argument('--wd', type=float, default=0.01)
parser.add_argument('--num_k', type=int, default=24)
parser.add_argument('--topk', type=int, default=12)
parser.add_argument('--epsilon', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=4e-5)
parser.add_argument('--lr_decay', type=int, default=10)
parser.add_argument('--lr_div', type=float, default=0.1)
parser.add_argument('--hy_regLoss', type=float, default=1.40)
parser.add_argument('--temperature', type=float, default=0.20)

args = parser.parse_args()


def load_Clip(clip_model_name):
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.eval().float()

    # Define the preprocess function
    if args.preprocess_type == "clip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used for training')
    elif args.preprocess_type == "targetpad":
        preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
        print(f'Target pad  preprocess pipeline is used for training')
    else:
        raise ValueError(f"preprocess_type should be either clip or targetpad, got {args.preprocess_type}")

    return clip_model, preprocess


def create_model_and_optimizer():
    clip_model, preprocess = load_Clip(args.clip_model_name)

    # load phi networks
    phi_s = Phi(input_dim=clip_model.visual.output_dim, hidden_dim=clip_model.visual.output_dim * 4,
                     output_dim=clip_model.token_embedding.embedding_dim, dropout=0.5)

    phi_a = Phi(input_dim=clip_model.visual.output_dim, hidden_dim=clip_model.visual.output_dim * 4,
                    output_dim=clip_model.token_embedding.embedding_dim, dropout=0.5)

    # load FTI4CIR network, namely img2text
    if args.clip_model_name == "ViT-L/14":
        img_dim = 1024
    elif args.clip_model_name == "ViT-B/32":
        img_dim = 768

    img2text = IMG2TEXT(img_patch_dim=img_dim, token_feat=clip_model.token_embedding.embedding_dim,
                        phi_s=phi_s, phi_a=phi_a, num_k=args.num_k, hy_regLoss=args.hy_regLoss,
                        temperature=args.temperature, tf_layer=args.tf_layer, tf_head=args.tf_head, topk=args.topk,
                        epsilon=args.epsilon).to(device, non_blocking=True)
    img2text = img2text.float()

    # define dataset and dataloader
    if args.pre_dataset == "ImageNetDataset":
        dataset = ImageNetDataset(args.imgCaptionPath, args.ImageNetPath, preprocess)
    else:
        raise ValueError(f"pre_dataset should be ImageNetPath, got {args.pre_dataset}")

    train_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=True)

    optimizer = optim.AdamW(img2text.parameters(), lr=args.lr, weight_decay=args.wd)

    # storage the hyperparameters
    with open(args.model_dir + '/hyperparameters.json', 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    return img2text, clip_model, train_dataloader, optimizer


def train(img2text, clip_model, optimizer, train_dataloader, scaler):
    img2text.train()
    loss_avg = RunningAverage()
    with tqdm(total=len(train_dataloader)) as t:
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            images = batch.get('image').cuda()
            subject = batch.get('subject')
            attribute = batch.get('attribute')

            with autocast():
                total_loss = img2text.getLoss(images, subject, attribute, clip_model)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_avg.update(total_loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()


def train_and_evaluate(img2text, clip_model, train_dataloader, optimizer):
    scaler = GradScaler()
    epoches = args.num_epochs

    for epoch in range(epoches):
        if epoch == args.lr_decay:
            for g in optimizer.param_groups:
                g['lr'] *= args.lr_div

        train(img2text, clip_model, optimizer, train_dataloader, scaler)

        if args.save_training:
            if epoch % args.save_frequency == 0:
                model_path = args.model_dir + "/save_model"
                submissions_folder_path = Path(model_path)
                submissions_folder_path.mkdir(exist_ok=True, parents=True)
                torch.save(img2text, os.path.join(model_path, f"model_{epoch}.pth"))


if __name__ == '__main__':
    # 种子固定
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    img2text, clip_model, train_dataloader, optimizer = create_model_and_optimizer()

    train_and_evaluate(img2text, clip_model, train_dataloader, optimizer)
