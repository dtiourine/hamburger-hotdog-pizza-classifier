import torch
import torch.nn as nn
import torchvision.models as models
import wandb
from tqdm import tqdm

def train_validate_model(num_epochs, train_loader, valid_loader, model, criterion, optimizer, device):
    wandb.watch(model, criterion, log="all", log_freq=10)

    best_train_accuracy = 0
    best_val_accuracy = 0

    # Create a single tqdm progress bar for the entire training cycle
    overall_progress_bar = tqdm(total=num_epochs, desc='Overall Training Progress', position=0, leave=True)

    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            wandb.log({"train_batch_loss": loss.item()})

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        best_train_accuracy = max(best_train_accuracy, train_accuracy)
        wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "epoch": epoch + 1})

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # Disable gradient calculation
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                wandb.log({"val_batch_loss": loss.item()})

        val_loss = val_running_loss / len(valid_loader)
        val_accuracy = 100 * val_correct / val_total
        best_val_accuracy = max(best_val_accuracy, val_accuracy)
        wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy, "epoch": epoch + 1})

        # Update the overall progress bar
        overall_progress_bar.set_postfix({
            'Best Train Accuracy': f'{best_train_accuracy:.2f}%',
            'Current Train Accuracy': f'{train_accuracy:.2f}%',
            'Best Val Accuracy': f'{best_val_accuracy:.2f}%',
            'Current Val Accuracy': f'{val_accuracy:.2f}%'
        })
        overall_progress_bar.update(1)

    overall_progress_bar.close()

def test_model(model, test_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Validation Accuracy: {100 * correct / total}%')


def modify_model_output(model_name, num_classes, device):
    """
    Modify the output layer of a pre-trained model to match the number of classes.

    Args:
    - model_name (str): Name of the model to modify ('alexnet', 'vgg16', 'resnet50', 'inception_v3').
    - num_classes (int): Number of output classes.

    Returns:
    - model (nn.Module): The modified model with the output layer matching the number of classes.
    """
    if model_name == 'alexnet':
        model = models.alexnet(weights='DEFAULT')
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name == 'vgg16':
        model = models.vgg16(weights='DEFAULT')
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'inception_v3':
        model = models.inception_v3(weights='DEFAULT')
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError("Unsupported model name. Choose from 'alexnet', 'vgg16', 'resnet50', 'inception_v3'.")

    return model.to(device)