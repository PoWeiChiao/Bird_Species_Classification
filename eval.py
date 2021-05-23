import glob
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from model import get_resnet50

def predict(net, device, image_path, image_transforms):
    image = Image.open(image_path)
    image = image_transforms(image)
    image = image.unsqueeze(0)
    image = image.to(device=device)

    net.eval()
    with torch.no_grad():
        pred = net(image)
        pred = np.array(pred.data.cpu()[0])
        return pred

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    net = get_resnet50(num_classes=265, pretrained=False)
    net.load_state_dict(torch.load('saved/model_best.pth', map_location=device))
    net.to(device=device)

    test_dir = 'data/test'
    species_list = os.listdir(test_dir)
    species_list.sort()

    images_list = []
    for i, species in enumerate(species_list):
        images = glob.glob(os.path.join(test_dir, species, '*.jpg'))
        for im in images:
            images_list.append([im, i])

    image_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    true_prediction = 0
    false_prediction = 0
    for image_data in images_list:
        pred = predict(net, device, image_data[0], image_transforms)
        prediction = np.argmax(pred)
        if prediction == image_data[1]:
            true_prediction += 1
        else:
            false_prediction += 1
            print(species_list[prediction])
            print(image_data[0])
    print('true prediction: {} false prediction: {}'.format(true_prediction, false_prediction))
    print('true ratio: {:.6f} false ratio: {:.6f}'.format(true_prediction / len(images_list), false_prediction / len(images_list)))

if __name__ == '__main__':
    main()