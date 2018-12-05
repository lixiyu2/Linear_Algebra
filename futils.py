import torchvision.models as models
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch
import PIL
from PIL import Image
import torch.nn.functional as F



# 加载模型
def setup(model='vgg16', dropout=0.5, hidden_layer1=512, lr=0.001, power='gpu'):

    if model == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model == 'densenet121':
        model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(dropout)),
        ('fc1', nn.Linear(25088, hidden_layer1)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_layer1, 120)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(120, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    if torch.cuda.is_available() and power == 'gpu':
        model.cuda()

    return model, criterion, optimizer


# 处理图片
def process_image(image_path):

    img = Image.open(image_path) # Here we open the image

    make_img_good = transforms.Compose([ # Here as we did with the traini ng data we will define a set of
        # transfomations that we will apply to the PIL image
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_image = make_img_good(img)

    return tensor_image

# 预测
def predict(image_path, model, topk=5,power='gpu'):
    if torch.cuda.is_available() and power=='gpu':
        model.to('cuda:0')

    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if power == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)

    probability = F.softmax(output.data,dim=1)

    return probability.topk(topk)