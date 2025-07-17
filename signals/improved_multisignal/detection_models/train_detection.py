import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime

from simple_detection_model import SimpleDetectionModel
from complex_detection_model import ComplexDetectionModel
from defect_focused_dataset import get_defect_focused_dataloader


def train_detection_model(model, train_loader, val_loader, num_epochs, device, model_name):
    """Simple detection training - no noise, clean signals only"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)
    criterion = nn.BCELoss()
    
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for signals, labels, _ in tqdm(train_loader, desc=f"Training {model_name}"):
            signals = signals.to(device)
            labels = labels.to(device)
            
            # Forward pass
            detection_prob = model(signals)
            loss = criterion(detection_prob, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            train_correct += ((detection_prob > 0.5) == (labels > 0.5)).sum().item()
            train_total += labels.numel()
        
        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for signals, labels, _ in tqdm(val_loader, desc="Validation"):
                signals = signals.to(device)
                labels = labels.to(device)
                
                detection_prob = model(signals)
                loss = criterion(detection_prob, labels)
                
                val_loss += loss.item()
                val_correct += ((detection_prob > 0.5) == (labels > 0.5)).sum().item()
                val_total += labels.numel()
        
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_accuracy:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_accuracy:.4f}")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), f"best_{model_name.lower()}_detection.pth")
            print(f"  New best accuracy: {val_accuracy:.4f}!")
        
        print()
    
    return best_accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader = get_defect_focused_dataloader(
        "json_data_07/", 
        batch_size=16, 
        seq_length=30,
        shuffle=True,
        validation_split=0.2,
        min_defects_per_sequence=1
    )
    
    # Test both models
    models = {
        "Simple": SimpleDetectionModel(signal_length=320),
        "Complex": ComplexDetectionModel(signal_length=320)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name} Detection Model")
        print(f"{'='*50}")
        
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        best_acc = train_detection_model(
            model, train_loader, val_loader, 
            num_epochs=15, device=device, model_name=name
        )
        
        results[name] = best_acc
        print(f"{name} Model Best Accuracy: {best_acc:.4f}")
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    for name, acc in results.items():
        print(f"{name} Detection Model: {acc:.4f}")


if __name__ == "__main__":
    main()
