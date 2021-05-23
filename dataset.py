import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BirdDataset(Dataset):
    def __init__(self, data_dir, image_transforms):
        self.data_dir = data_dir
        self.image_transforms = image_transforms

        species_list = os.listdir(data_dir)
        species_list.sort()

        self.images_list = []
        for i, species in enumerate(species_list):
            images = glob.glob(os.path.join(data_dir, species, '*.jpg'))
            for im in images:
                self.images_list.append([im, i])

        self.images_list.sort()

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image = Image.open(self.images_list[index][0])
        image = self.image_transforms(image)

        label = torch.as_tensor(self.images_list[index][1], dtype=torch.long)
        return image, label

def main():
    data_dir = 'data/train'
    image_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = BirdDataset(data_dir, image_transforms)
    print(dataset.__len__())

if __name__ == '__main__':
    main()