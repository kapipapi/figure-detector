import string

import torch
from PIL import Image
from torch import nn
from torchvision import transforms


class LetterModel:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, weights_path):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(string.ascii_letters))

        self.model.load_state_dict(torch.load(weights_path))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

    def classify_letter(self, img):
        img = Image.fromarray(img)
        img = self.preprocess(img)
        img = torch.unsqueeze(img, 0)

        img = img.to(self.device)

        outputs = self.model(img)

        return string.ascii_letters[outputs.cpu().detach().numpy().argmax()]
