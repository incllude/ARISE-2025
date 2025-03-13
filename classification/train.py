import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from accelerate import Accelerator
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import wandb
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tqdm import tqdm
from metrics import calculate_accuracy
from data_utils.datasets import initialize_data
from classification.utils import save_checkpoint
from accelerate.utils import set_seed
import torch.nn.functional as F
import monai


set_seed(43)
monai.utils.set_determinism(seed=43)
accelerator = Accelerator()


class FocalLoss(nn.Module):
    
    def __init__(self, gamma=2, weights=None):
        super().__init__()
        
        self.gamma = gamma
        self.weights = weights
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        weights = self.weights.to(inputs.device).gather(0, targets.view(-1))
        loss = weights * (1-pt)**self.gamma * ce_loss

        return loss.sum() / weights.sum()


def train_epoch(cfg, model, train_loader, optimizer, criterion_erosion, criterion_jsn, accelerator):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        img, erosion_score, jsn_score = batch
        img = img.double()
        
        outputs = model(img)
        
        loss_jsn = criterion_jsn(outputs["jsn"], jsn_score)
        loss_erosion = criterion_erosion(outputs["erosion"], erosion_score)
        loss = loss_jsn + loss_erosion

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()

        # Log to WandB
        if accelerator.is_local_main_process:
            wandb.log({
                "train_loss": loss.item(),
                "train_loss_jsn": loss_jsn.item(),
                "train_loss_erosion": loss_erosion.item(),
                "lr": optimizer.param_groups[0]["lr"]
            })

    return total_loss / len(train_loader)

# Validation function
def validate_epoch(cfg, model, val_loader, criterion_erosion, criterion_jsn, accelerator):
    model.eval()
    total_loss = 0.0
    jsn_accuracy, erosion_accuracy = 0.0, 0.0

    jsn_res = []
    jsn_labels = []
    erosion_res = []
    erosion_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            img, erosion_score, jsn_score = batch
            img = img.double()
            
            outputs = model(img)
            
            loss_jsn = criterion_jsn(outputs["jsn"], jsn_score)
            loss_erosion = criterion_erosion(outputs["erosion"], erosion_score)
            loss = loss_jsn + loss_erosion

            total_loss += loss.item()
            
            jsn_res.extend(outputs["jsn"].detach().cpu().argmax(dim=1).tolist())
            jsn_labels.extend(jsn_score.detach().cpu().tolist())
            
            erosion_res.extend(outputs["erosion"].detach().cpu().argmax(dim=1).tolist())
            erosion_labels.extend(erosion_score.detach().cpu().tolist())
            
            # Calculate weighted-accuracy
        jsn_accuracy = balanced_accuracy_score(jsn_labels, jsn_res)
        erosion_accuracy = balanced_accuracy_score(erosion_labels, erosion_res)

    avg_loss = total_loss / len(val_loader)

    # Log to WandB
    if accelerator.is_local_main_process:
        wandb.log({
            "val_loss": avg_loss,
            "val_jsn_accuracy": jsn_accuracy,
            "val_erosion_accuracy": erosion_accuracy
        })

    return avg_loss, jsn_accuracy, erosion_accuracy


@hydra.main(config_path="../config/classification", config_name="train")
def main(cfg: DictConfig):

    if accelerator.is_local_main_process:
        os.environ["WANDB_API_KEY"] = cfg.wandb.api_key
        wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True))

    model = instantiate(cfg.model).double()

    
    # optimizer = instantiate(cfg.optimizer, params=model.parameters())
    optimizer = eval(cfg.optimizer._target_)([
        {"params": model.heads.parameters(), "lr": cfg.optimizer.head_lr, "weight_decay": cfg.optimizer.weight_decay},
        {"params": model.backbone.parameters(), "lr": cfg.optimizer.lr, "weight_decay": cfg.optimizer.weight_decay}
    ])
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    train_loader, val_loader = initialize_data(cfg)

    criterion_erosion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.training.normalized_erosion_class_weights).double().cuda())
    criterion_jsn = nn.CrossEntropyLoss(weight=torch.tensor(cfg.training.normalized_jsn_class_weights).double().cuda())
    
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Training loop
    for epoch in range(cfg.training.epochs):
        train_loss = train_epoch(cfg, model, train_loader, optimizer, criterion_erosion, criterion_jsn, accelerator)
        val_loss, jsn_accuracy, erosion_accuracy = validate_epoch(cfg, model, val_loader, criterion_erosion, criterion_jsn, accelerator)

        # Step the scheduler
        scheduler.step()

        # Print epoch results
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch + 1}/{cfg.training.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"JSN Accuracy: {jsn_accuracy:.4f}")
            print(f"Erosion Accuracy: {erosion_accuracy:.4f}")
            
        if accelerator.is_local_main_process:
            save_checkpoint(model, epoch + 1, jsn_accuracy, erosion_accuracy, cfg.training.save_dir)


if __name__ == "__main__":
    main()