import torch
from torch import nn
import numpy as np
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Literal, Union
import torch.nn.functional as F
import os
from tqdm import tqdm
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser, BooleanOptionalAction
from torch.utils.data import TensorDataset

def step(model: nn.Module,
         optimizer: Optimizer,
         loader: DataLoader,
         writer: SummaryWriter,
         mode: Union[Literal["train"], Literal["val"], Literal["test"]],
         epoch: int,
         device: str):
    assert mode in ["train", "val", "test"]
    if mode == "train":
        model = model.train()
    else:
        model = model.eval()
    
    losses = []

    for i, (x, y) in tqdm(enumerate(loader), f"Epoch {epoch}"):
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(mode == "train"):
            output = model(x)
        loss = F.mse_loss(output, y)
        rmse = loss.sqrt()

        if mode == "train":
            writer.add_scalar(f"{mode}/loss", loss.detach().item(), epoch*len(loader) + i)
            writer.add_scalar(f"{mode}/rmse", rmse.detach().item(), epoch*len(loader) + i)
        losses.append(loss.detach().item())

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    avg_loss = torch.tensor(losses).mean().detach()
    if mode != "train":
        writer.add_scalar(f"{mode}/loss", avg_loss.item(), epoch)
        writer.add_scalar(f"{mode}/rmse", avg_loss.sqrt().item(), epoch)
    return avg_loss


@torch.no_grad()
def knn_classifier(train_features: torch.Tensor,
                    train_labels: torch.Tensor,
                    test_features: torch.Tensor,
                    test_labels: torch.Tensor,
                    k: int,
                    device: str):
    train_features = train_features.t().to(device)
    train_labels = train_labels.squeeze().to(device)
    num_test_samples = test_labels.shape[0]
    num_chunks = 1000
    samples_per_chunk = num_test_samples // num_chunks
    preds = []
    for idx in tqdm(range(0, num_test_samples, samples_per_chunk), "[KNN] regression"):
        features = test_features[idx : idx + samples_per_chunk].to(device)
        targets = test_labels[idx : idx + samples_per_chunk].to(device)
        batch_size = targets.shape[0]

        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.unsqueeze(0).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)
        weights = distances / distances.sum(dim=-1, keepdim=True)
        pred = retrieved_neighbors.unsqueeze(-2) @ weights.unsqueeze(-1)
        preds.append(pred.squeeze().cpu())
        torch.cuda.empty_cache()
    preds = torch.cat(preds, dim=0).reshape(test_labels.shape)
    preds = preds.to(device)
    test_labels = test_labels.to(device)
    loss = F.mse_loss(test_labels, preds)
    return loss



def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--train_target_path", type=str, required=True)
    parser.add_argument("--val_target_path", type=str, required=True)
    parser.add_argument("--test_target_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)    
    parser.add_argument("--evals", nargs="+")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--delta_treshold", type=float, default=0.00001)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")  
    return parser

def eval_linear(args, train_loader, val_loader, test_loader, writer, ckpt_dir):
    D = next(iter(train_loader))[0].shape[-1]
    model = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))
    optimizer = SGD(model.parameters(), momentum=0.9, weight_decay=0, lr=args.lr * args.batch_size / 256)
    model = model.to(args.device)
    best_val_score = 1e8
    patience_rest = args.patience

    for epoch in tqdm(range(args.epochs), "Training regression head"):
        if patience_rest <= 0:
            break

        step(model, optimizer, train_loader, writer, "train", epoch, args.device)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last_ckpt.pt"))
            
        val_score = step(model, optimizer, val_loader, writer, "val", epoch, args.device)
        delta = best_val_score - val_score
        if val_score < best_val_score:
            print(f"New best validation score: {val_score}, delta: {delta}")
            best_val_score = val_score
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_ckpt.pt"))
        
        if delta >= args.delta_treshold:
            patience_rest = args.patience
        else:
            patience_rest -= 1
            print(f"delta < {args.delta_treshold}, patience_rest: {patience_rest}")

        loss = step(model, optimizer, test_loader, writer, "test", epoch, args.device)
    return {
        "mse": loss.item(),
        "rmse": loss.sqrt().item(),
    }

def eval_knn(args, train_dataset: TensorDataset, val_dataset: TensorDataset, test_dataset: TensorDataset):
    best_loss = 1e8
    best_k = 3
    for k in tqdm([5,10,20,100,200], "[KNN] Choosing best k"):
        loss = knn_classifier(train_dataset.data, train_dataset.target, val_dataset.data, val_dataset.target, k, args.device)
        print(f"KNN: k={k}, mse={loss.item()}")
        if loss < best_loss:
            best_loss = loss
            best_k = k
        
    loss = knn_classifier(train_dataset.data, train_dataset.target, test_dataset.data, test_dataset.target, best_k, args.device)
    return {
            "best_k": best_k,
            "mse": loss.item(),
            "rmse": loss.sqrt().item(),
        }
    
def main(args):
        
    train_dataset = TensorDataset(torch.from_numpy(np.load(args.train_data_path)), torch.from_numpy(np.load(args.train_target_path)))
    val_dataset = TensorDataset(torch.from_numpy(np.load(args.val_data_path)), torch.from_numpy(np.load(args.val_target_path)))
    test_dataset = TensorDataset(torch.from_numpy(np.load(args.test_data_path)), torch.from_numpy(np.load(args.test_target_path)))
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)


    for eval_str in args.evals:
        out_dir = os.path.join(args.output_dir, eval_str, args.target, "_".join(str(datetime.now()).split()))
        os.makedirs(out_dir, exist_ok=False)
        if eval_str == "eval_linear":
            writer = SummaryWriter(out_dir)
            ret = eval_linear(args, train_loader, val_loader, test_loader, writer, out_dir)
        elif eval_str == "eval_knn":
            ret = eval_knn(args, train_dataset, val_dataset, test_dataset)
        else:
            assert False, f"`{eval_str}` not found"
        
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(ret, f)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)