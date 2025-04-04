import torch
import torch.nn as nn
import os
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def train_model(model, train_loader, valid_loader, task, optimizer,
                num_epochs, device, output_dir='results'):
    
    writer = SummaryWriter(log_dir=f'{output_dir}/runs')
    model.to(device)

    if task == 'age':
        criterion = nn.L1Loss()
    elif task == 'sex':
        criterion = nn.BCELoss()
    elif task == 'cognitive_status':
        criterion = nn.CrossEntropyLoss()
    
    else:
        raise ValueError(f'Invalid task value: {task}.')

    for epoch in range(num_epochs):
        model.train()
        total_loss_train = 0.0
        predictions, ground_truths = [], []
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            if task != 'cognitive_status':
                outputs = outputs.squeeze(1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            if task == 'cognitive_status':
                outputs_extend = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            elif task == 'sex':
                outputs_extend = (outputs.detach().cpu().numpy() > 0.5).astype(int)
            else:
                outputs_extend = outputs.detach().cpu().numpy()
            predictions.extend(outputs_extend)
            ground_truths.extend(targets.cpu().numpy())

        avg_loss_train = total_loss_train / len(train_loader)
        if task in ['sex', 'cognitive_status']:
            eval_train = accuracy_score(ground_truths, predictions)
        else:
            eval_train = mean_absolute_error(ground_truths, predictions)
            # eval_train = (abs(predictions - ground_truths) / ground_truths).mean()

        print(f"==TRAIN== Epoch {epoch+1}, Task {task}, Loss={avg_loss_train:.4f}, Eval={eval_train:.4f}...")

        # === VALIDATION ===
        model.eval()
        total_loss_valid = 0.0
        predictions, ground_truths = [], []
        with torch.no_grad():
            for features, targets in valid_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                if task != 'cognitive_status':
                    outputs = outputs.squeeze(1)

                loss = criterion(outputs, targets)

                total_loss_valid += loss.item()
                if task == 'cognitive_status':
                    outputs_extend = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                elif task == 'sex':
                    outputs_extend = (outputs.detach().cpu().numpy() > 0.5).astype(int)
                else:
                    outputs_extend = outputs.detach().cpu().numpy()
                predictions.extend(outputs_extend)
                ground_truths.extend(targets.cpu().numpy())

            avg_loss_valid = total_loss_valid / len(valid_loader)

            if task in ['sex', 'cognitive_status']:
                eval_valid = accuracy_score(ground_truths, predictions)
            else:
                eval_valid = mean_absolute_error(ground_truths, predictions)
                # eval_valid = (abs(predictions - ground_truths) / ground_truths).mean()
        print(f"==VALID== Epoch {epoch+1}, Task {task}, Loss={avg_loss_valid:.4f}, Eval={eval_valid:.4f}...")
        # === METRIC ===
        writer.add_scalars(f'Loss/1-total_loss_{task}', {'train': avg_loss_train, 'test': avg_loss_valid}, epoch)
        writer.add_scalars(f'Loss/2-metric_{task}', {'train': eval_train, 'test': eval_valid}, epoch)
        # confusion matrix
        if task in ['sex', 'cognitive_status']:
            class_names = ['Female', 'Male'] if task == 'sex' else ['Normal', 'MCI', 'AD']
            cm_title = f"{task.capitalize()} Ep{epoch+1} (Acc {eval_valid:.2f})"
            cm_path = f"{output_dir}/confusion_matrix/{task}_epoch_{epoch+1}.png"

            save_confusion_matrix(
                y_true=ground_truths,
                y_pred=predictions,
                labels=class_names,
                title=cm_title,
                save_path=cm_path,
                normalize='true'
            )

    writer.close()


def value_check(config):
    task = config['task']
    if task not in ['age', 'sex', 'cognitive_status']:
        raise ValueError(f'Unknown task: {task}. ' + \
                        'Choose one from "age", "sex", "cognitive_status".')
    

def save_confusion_matrix(y_true, y_pred, labels, title, save_path, normalize=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)), normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format=".2f" if normalize else "d")
    if normalize:
        disp.im_.set_clim(0, 1) 
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
