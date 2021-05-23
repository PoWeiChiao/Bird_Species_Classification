import os
from torchvision.transforms.transforms import ColorJitter, RandomRotation
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import BirdDataset
from logger import Logger
from model import get_resnet50

def train(net, device, dataset_train, dataset_val, batch_size=16, epochs=100, lr=1e-3):
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True)

    optimizer = optim.SGD(params=net.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.7, patience=4)

    train_log = Logger('saved/train_log.txt')
    val_log = Logger('saved/val_log.txt')
    best_loss = float('inf')
    writer = SummaryWriter('runs/exp_4')

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        print('running epoch {}'.format(epoch))

        net.train()
        for images, labels in tqdm(train_loader):
            images = images.to(device=device)
            labels = labels.to(device=device)

            pred = net(images)
            loss = criterion(pred, labels)
            train_loss += loss.item() * images.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        for images, labels in tqdm(val_loader):
            images = images.to(device=device)
            labels = labels.to(device=device)

            with torch.no_grad():
                pred = net(images)
                loss = criterion(pred, labels)
                val_loss += loss.item() * images.size(0)

        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, val_loss))
        writer.add_scalar(tag='train loss', scalar_value=train_loss, global_step=epoch)
        writer.add_scalar(tag='val loss', scalar_value=val_loss, global_step=epoch)
        train_log.write_epoch_loss(epoch, train_loss)
        val_log.write_epoch_loss(epoch, val_loss)

        lr_scheduler.step(val_loss)

        if epoch >= 80:
            torch.save(net.state_dict(), 'saved/model_{}.pth'.format(epoch))
        if val_loss <= best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), 'saved/model_best.pth')
            print('model saved')
    writer.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    net = get_resnet50(num_classes=265, pretrained=False)
    if os.path.isfile('saved/model_best.pth'):
        net.load_state_dict(torch.load('saved/model_best.pth', map_location=device))
    net.to(device=device)

    train_dir = 'data/train'
    val_dir = 'data/val'
    train_image_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=30),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_image_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_train = BirdDataset(data_dir=train_dir, image_transforms=train_image_transforms)
    dataset_val = BirdDataset(data_dir=val_dir, image_transforms=val_image_transforms)

    train(net, device, dataset_train, dataset_val)

if __name__ == '__main__':
    main()
