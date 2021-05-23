import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet50(num_classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    num_output = model.fc.in_features
    model.fc = nn.Linear(num_output, num_classes)
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_resnet50(num_classes=256, pretrained=False)
    model.to(device=device)

    input = torch.rand(1, 3, 224, 224)
    input = input.to(device=device)

    pred = model(input)
    print(pred)

if __name__ == '__main__':
    main()