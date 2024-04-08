
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import splitfolders



DATASET_DIR = r"C:\Users\Reema Reny\Documents\GitHub\COMP6721-GroupB\Rice Leaf Disease Images\Rice Leaf Disease Images"


splitfolders.ratio(r"C:\Users\Reema Reny\Documents\GitHub\COMP6721-GroupB\Rice Leaf Disease Images\Rice Leaf Disease Images", output=r"C:\Users\Reema Reny\Documents\GitHub\COMP6721-GroupB\Rice Leaf Disease Images\Rice Leaf Disease Images", seed=1337, ratio=(0.8, 0.1, 0.1))

train_dir = DATASET_DIR + "/train"
valid_dir = DATASET_DIR + "/val"
diseases = os.listdir(train_dir)

diseases = [item for item in diseases if item not in {'train', 'val', 'test'}]

# Print dieases name
print(diseases)

print(f"Total classes are: {len(diseases)}")

nums = {disease: len(os.listdir(os.path.join(train_dir, disease))) for disease in diseases}

# Converting the nums dictionary to pandas DataFrame with plant name as index and number of images as column
img_per_class = pd.DataFrame(nums.values(), index=nums.keys(), columns=["No. of images"])

img_per_class

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomPerspective(distortion_scale=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def is_valid_file(file_path):
    try:
        # Open the image file
        with open(file_path, 'rb') as f:
            img = Image.open(f)

            # Attempt to load the image to ensure it's valid
            img.load()
             # Verify the integrity of the image file
            img.verify()
            return True
    except Exception as e:
        # If any exception occurs during the process, consider the file as invalid
        return False


batch_size=32

train_dataset = ImageFolder(root=r"C:\Users\Reema Reny\Documents\GitHub\COMP6721-GroupB\Rice Leaf Disease Images\Rice Leaf Disease Images\train", transform=train_transforms,is_valid_file=is_valid_file)
val_dataset = ImageFolder(root=r"C:\Users\Reema Reny\Documents\GitHub\COMP6721-GroupB\Rice Leaf Disease Images\Rice Leaf Disease Images\val", transform=val_transforms,is_valid_file=is_valid_file)
test_dataset = ImageFolder(root=r"C:\Users\Reema Reny\Documents\GitHub\COMP6721-GroupB\Rice Leaf Disease Images\Rice Leaf Disease Images\test", transform=val_transforms,is_valid_file=is_valid_file)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

unique_classes = set()
class_to_idx = train_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

num_classes = len(class_to_idx)
num_cols = min(5, num_classes)
num_rows = math.ceil(num_classes / num_cols)

if num_rows == 1:
    fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(15, 5))
else:
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, num_rows * 3))

row_index = 0
col_index = 0

for images, labels in train_loader:
    for image, label in zip(images, labels):
        class_name = idx_to_class[label.item()]
        if class_name not in unique_classes:
            unique_classes.add(class_name)
            if num_rows == 1:
                ax = axes[col_index]
            else:
                ax = axes[row_index, col_index]
            img = image.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            ax.imshow(img)
            ax.set_title(class_name)
            ax.axis('off')

            col_index += 1
            if col_index == num_cols:
                col_index = 0
                row_index += 1

        if len(unique_classes) == num_classes:
            break
    if len(unique_classes) == num_classes:
        break

plt.tight_layout()
plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')
model = models.vgg19(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Sequential(
#     nn.Dropout(0.7),  # Increased dropout rate
#     nn.Linear(num_ftrs, len(train_dataset.classes))
# )
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
    nn.Dropout(0.7),
    nn.Linear(num_ftrs, len(train_dataset.classes))
)
model.to(device)



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

TENSORBOARD_SUMMARY_PATH=f'{DATASET_DIR}/Rice_Leaf_Disease_Images'
writer = SummaryWriter(TENSORBOARD_SUMMARY_PATH)

BEST_MODEL_PATH=r"C:\Users\Reema Reny\Documents\GitHub\COMP6721-GroupB\Rice Leaf Disease Images\Rice Leaf Disease Images\Rice_Leaf_Disease_Images.pth"

early_stopping_patience = 5
epochs_no_improve = 0
val_loss_min = np.Inf

def train_model(num_epochs):
    global epochs_no_improve
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Train the model
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        writer.add_scalar('Training loss', epoch_loss, epoch)
        writer.add_scalar('Training accuracy', epoch_acc, epoch)


        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / val_total
        val_accuracy = val_correct / val_total
        writer.add_scalar('Validation loss', val_loss, epoch)
        writer.add_scalar('Validation accuracy', val_accuracy, epoch)

        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            epochs_no_improve += 1

        if epochs_no_improve == early_stopping_patience:
            print('Early stopping initiated...')
            break

    #to save the training data
    torch.save(model.state_dict(), 'model_VGG_Rice.ckpt')


    # Model evaluation- use the below line if u already have the training data and do not want to run the epochs again
    # model.load_state_dict(torch.load('model_VGG_Rice.ckpt'))
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    test_correct = 0
    test_total = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_accuracy = test_correct / test_total
    print(f'\n\nTest Accuracy: {test_accuracy:.4f}')

    writer.close()

    conf_matrix = confusion_matrix(all_targets, all_predictions)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    class_report = classification_report(all_targets, all_predictions)
    print("Classification Report:")
    print(class_report)

    
def extract_features(loader, model, device):
        features = []
        targets = []
        model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                features.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        return np.array(features), np.array(targets)

def load_tensorboard_metrics(summary_path):
    event_acc = EventAccumulator(summary_path)
    event_acc.Reload()

    train_loss = [scalar.value for scalar in event_acc.Scalars('Training loss')]
    val_loss = [scalar.value for scalar in event_acc.Scalars('Validation loss')]
    train_accuracy = [scalar.value for scalar in event_acc.Scalars('Training accuracy')]
    val_accuracy = [scalar.value for scalar in event_acc.Scalars('Validation accuracy')]

    return train_loss, train_accuracy, val_loss, val_accuracy

def plot_metrics(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(1, len(train_loss) + 1), y=train_loss, label='Training Loss')
    sns.lineplot(x=range(1, len(val_loss) + 1), y=val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def plot_metrics2(train_accuracy, val_accuracy):  
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(1, len(train_accuracy) + 1), y=train_accuracy, label='Training Accuracy')
    sns.lineplot(x=range(1, len(val_accuracy) + 1), y=val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def main():
    # Load the model checkpoint
    checkpoint_path = 'model_VGG_Rice.ckpt'
    if os.path.exists(checkpoint_path):
        print("Loading the model checkpoint...")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("Checkpoint file not found.")
        return

    if os.path.exists(TENSORBOARD_SUMMARY_PATH):
        train_loss, train_accuracy, val_loss, val_accuracy = load_tensorboard_metrics(TENSORBOARD_SUMMARY_PATH)
        plot_metrics(train_loss, val_loss)
        plot_metrics2(train_accuracy,val_accuracy)

    # Evaluating the model
    # accuracy = evaluate_model(model, test_loader)
    # print(f"Test accuracy: {accuracy:.2f}%")

    train_features, train_targets = extract_features(train_loader, model, device)
    val_features, val_targets = extract_features(val_loader, model, device)

    # Combine train and validation features and targets for t-SNE
    all_features = np.concatenate((train_features, val_features), axis=0)
    all_targets = np.concatenate((train_targets, val_targets), axis=0)

    # t-SNE transformation
    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(all_features)

    # Visualization
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=all_targets, palette='tab20', legend='full')
    plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

if __name__ == '__main__':
    num_epochs = 10
    train_model(num_epochs)
    main()
