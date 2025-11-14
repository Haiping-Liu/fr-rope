import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import wandb
import os
from tqdm import tqdm

from model import SpatialTransformer
from dataset import STDataset
from fr import compute_axial_rope, compute_fr_rope


def train_epoch(model, dataloader, optimizer, criterion, device, rope_type='axial'):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc='Training')
    for coords, genes, labels in pbar:
        coords = coords.to(device)
        genes = genes.to(device)
        labels = labels.to(device)

        coords = coords.squeeze(0)
        genes = genes.squeeze(0)
        labels = labels.squeeze(0)

        head_dim = model.dim // model.encoder[0].attn.num_heads
        if rope_type == 'axial':
            freqs_cis = compute_axial_rope(dim=head_dim, X=coords)
        elif rope_type == 'fr':
            freqs_cis = compute_fr_rope(dim=head_dim, X=coords)
        else:
            raise ValueError(f"Unknown rope_type: {rope_type}")

        genes = genes.unsqueeze(0)

        optimizer.zero_grad()
        logits = model(genes, freqs_cis)
        logits = logits.squeeze(0)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=-1)
        correct = (pred == labels).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_samples += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct / labels.size(0):.4f}'})

    return total_loss / len(dataloader), total_correct / total_samples


def evaluate(model, dataloader, criterion, device, rope_type='axial'):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for coords, genes, labels in pbar:
            coords = coords.to(device)
            genes = genes.to(device)
            labels = labels.to(device)

            coords = coords.squeeze(0)
            genes = genes.squeeze(0)
            labels = labels.squeeze(0)

            head_dim = model.dim // model.encoder[0].attn.num_heads
            if rope_type == 'axial':
                freqs_cis = compute_axial_rope(dim=head_dim, X=coords)
            elif rope_type == 'fr':
                freqs_cis = compute_fr_rope(dim=head_dim, X=coords)
            else:
                raise ValueError(f"Unknown rope_type: {rope_type}")

            genes = genes.unsqueeze(0)

            logits = model(genes, freqs_cis)
            logits = logits.squeeze(0)

            loss = criterion(logits, labels)

            pred = logits.argmax(dim=-1)
            correct = (pred == labels).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct / labels.size(0):.4f}'})

    return total_loss / len(dataloader), total_correct / total_samples


def main(args):
    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = STDataset(data_dir=args.data_dir, max_cells=args.max_cells, target_sum=args.target_sum)

    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    sample_coords, sample_genes, sample_labels = dataset[0]
    gene_dim = sample_genes.shape[1]
    num_classes = len(torch.unique(sample_labels))
    print(f"Gene dim: {gene_dim}, Num classes: {num_classes}")

    model = SpatialTransformer(
        gene_dim=gene_dim,
        dim=args.dim,
        num_heads=args.num_heads,
        n_layers=args.n_layers,
        num_classes=num_classes,
        qkv_bias=True,
        drop=args.dropout,
        attn_drop=args.attn_drop
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, args.rope_type)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, args.rope_type)

        scheduler.step()

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

            os.makedirs(args.save_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.save_dir, f'{args.run_name}_best.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model: {checkpoint_path}")

            wandb.run.summary['best_val_acc'] = best_val_acc
            wandb.run.summary['best_epoch'] = best_epoch

    print(f"\nBest Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Mouse_hypothalamic')
    parser.add_argument('--max_cells', type=int, default=2048)
    parser.add_argument('--target_sum', type=float, default=1e4)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attn_drop', type=float, default=0.1)
    parser.add_argument('--rope_type', type=str, default='axial', choices=['axial', 'fr'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb_project', type=str, default='fr-rope')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints')

    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f'{args.rope_type}_dim{args.dim}_layers{args.n_layers}'

    main(args)
