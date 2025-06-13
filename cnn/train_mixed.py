import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, accuracy_score
from tqdm import tqdm
import shutil

def prepare_mixed_data(real_data_dir, synthetic_normal_dir, synthetic_pneumonia_dir, dest_dir):
    """Prepares a mixed dataset of real and synthetic images."""
    if os.path.exists(dest_dir):
        print(f"Destination directory {dest_dir} already exists. Skipping data preparation.")
        return

    print(f"Creating directory structure in {dest_dir}...")
    
    def copy_files(source_dir, dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        for item in tqdm(os.listdir(source_dir), desc=f"Copying from {os.path.basename(source_dir)}"):
            s = os.path.join(source_dir, item)
            d = os.path.join(dest_dir, item)
            if os.path.isfile(s):
                shutil.copy2(s, d)

    print("Copying real data...")
    for split in ['train', 'val', 'test']:
        for class_name in ['NORMAL', 'PNEUMONIA']:
            real_source_path = os.path.join(real_data_dir, split, class_name)
            if os.path.isdir(real_source_path):
                dest_path = os.path.join(dest_dir, split, class_name)
                copy_files(real_source_path, dest_path)

    print("Augmenting training data with synthetic images...")
    copy_files(synthetic_normal_dir, os.path.join(dest_dir, 'train', 'NORMAL'))
    copy_files(synthetic_pneumonia_dir, os.path.join(dest_dir, 'train', 'PNEUMONIA'))
    
    print("Data preparation complete.")


def plot_confusion_matrix(cm, class_names, file_name='confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(file_name)
    plt.close()

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', pos_label=1)
    auc = roc_auc_score(all_labels, all_probs)
    
    print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')

    cm = confusion_matrix(all_labels, all_preds)
    return cm, {'loss': loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}

def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=50):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    class_names = val_loader.dataset.classes

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}' + '\n' + '-' * 10)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            dataloader = train_loader if phase == 'train' else val_loader
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs - 1} - {phase}"):
                inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = (torch.sigmoid(outputs) > 0.5).int()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                cm, metrics = evaluate_model(model, val_loader, criterion, device)
                if metrics['accuracy'] > best_acc:
                    best_acc = metrics['accuracy']
                    best_model_wts = model.state_dict()
                    torch.save(model.state_dict(), 'mixed_dit_best_validation_model.pth')
                    plot_confusion_matrix(cm, class_names, 'mixed_dit_best_confusion_matrix.png')

    print('Training complete. Saving final model state.')
    torch.save(model.state_dict(), 'mixed_dit_final_model.pth')
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    prepare_mixed_data(
        real_data_dir='pneumonia_data/chest_xray',
        synthetic_normal_dir='Archive/img_dit_normal',
        synthetic_pneumonia_dir='Archive/img_dit_pneumonia',
        dest_dir='mixed_dit_data_for_training'
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    data_dir = 'mixed_dit_data_for_training'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4 if device.type == 'cuda' else 0)
                   for x in ['train', 'val']}
    dataloaders['test'] = DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=4 if device.type == 'cuda' else 0)

    class_names = image_datasets['train'].classes
    print(f"Classes: {class_names}")

    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001)

    trained_model = train_model(model, criterion, optimizer, dataloaders['train'], dataloaders['val'], device, num_epochs=50)

    print("\nEvaluating on test set...")
    test_cm, test_metrics = evaluate_model(trained_model, dataloaders['test'], criterion, device)
    plot_confusion_matrix(test_cm, class_names, 'mixed_dit_test_confusion_matrix.png')
    print("Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}") 