import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from models.classifier import ClassifierPCAM


class WrapperDataset(Dataset):
    """Wrapper dataset to put into a dataloader."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def show(img):
    img = img.view(img.shape[1], img.shape[2], img.shape[0])
    print(type(img), img.shape)

    plt.imshow(img, cmap="gray")
    plt.show()


def read_in_images(filepath):
    image_dict = {}

    counter = 0
    transform = ToTensor()
    for filename in glob.glob(f"{filepath}/*.png"):
        if counter < 10:
            im = Image.open(filename)

            newsize = (128, 128)
            im = im.resize(newsize)
            im = ImageOps.grayscale(im)
            im = transform(im)

            name = filename.split(f"{filepath}/")[1].split(".png")[0]
            image_dict[name] = im
            counter += 1
        else:
            break

    return image_dict


def map_x_to_y(img_dict, label_df):
    X = []
    y = []

    for key in img_dict.keys():
        diagnosis = label_df[label_df["id_code"] == key]["diagnosis"]
        y_val = 1 if int(diagnosis) > 0 else 0
        X_val = img_dict.get(key)

        X.append(X_val)
        y.append(y_val)

    return X, y


if __name__ == "__main__":
    num_channels = 1
    num_classes = 2
    epochs = 10
    amt_of_data = 1.0
    train_test_split = 0.8

    data_dir = f"data/blindness_detection"

    train_images = read_in_images(f"{data_dir}/train_images")
    train_labels = pd.read_csv(f"{data_dir}/train.csv")
    X, y = map_x_to_y(train_images, train_labels)

    show(X[0])
    print(y[0])

    show(X[9])
    print(y[9])

    train_X = X[: int(len(train_images) * train_test_split * amt_of_data)]
    test_X = X[
        int(len(train_images) * train_test_split * amt_of_data) : int(
            len(train_images) * amt_of_data
        )
    ]

    train_y = y[: int(len(train_images) * train_test_split * amt_of_data)]
    test_y = y[
        int(len(train_images) * train_test_split * amt_of_data) : int(
            len(train_images) * amt_of_data
        )
    ]

    train_dataset = WrapperDataset(train_X, train_y)
    test_dataset = WrapperDataset(test_X, test_y)

    print(len(train_dataset), len(test_dataset))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=32)

    classifier = ClassifierPCAM(num_channels, num_classes)
    loss_func = CrossEntropyLoss()
    optimizer = Adam(classifier.parameters(), lr=0.001)

    for e in range(epochs):
        for batch_idx, (X_batch, y_batch) in enumerate(train_dataloader):
            # Forward pass through model
            output = classifier(X_batch)

            # Compute loss with pre-defined loss function
            loss = loss_func(output, y_batch)

            # Gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        classifier.eval()

        total_correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(test_dataloader):
            # Forward pass through model
            test_logits = classifier(X_batch)
            pred_probs = F.softmax(input=test_logits, dim=1)
            y_pred = torch.argmax(pred_probs, dim=1)
            total_correct += np.sum((y_pred == y_batch).numpy())

        accuracy = round(total_correct / len(test_dataloader.dataset) * 100, 2)
        print(f"Model accuracy was: {accuracy}%")
