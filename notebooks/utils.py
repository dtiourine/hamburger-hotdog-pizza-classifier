import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

def train_validate_model(num_epochs, train_loader, valid_loader, model, criterion, optimizer, device,
                         model_save_path):

    best_val_accuracy = 0

    overall_progress_bar = tqdm(total=num_epochs, desc='Overall Training Progress', position=0, leave=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(valid_loader)
        val_accuracy = 100 * val_correct / val_total

        # Save the model if the validation accuracy is the best we've seen so far.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model, model_save_path)

        overall_progress_bar.set_postfix({
            'Best Val Accuracy': f'{best_val_accuracy:.2f}%',
            'Current Train Accuracy': f'{train_accuracy:.2f}%',
            'Current Val Accuracy': f'{val_accuracy:.2f}%'
        })
        overall_progress_bar.update(1)

    overall_progress_bar.close()

def test_model(model, test_loader, device):
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
        print(f'Test Accuracy: {(100 * correct / total):.2f}%')


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