import argparse
import torch
import torchvision.models as models
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import futils

ap = argparse.ArgumentParser(description='Train.py')
# Command Line ardguments

ap.add_argument('data_dir', nargs='*', action="store", default="flowers")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest="dropout", action="store", default=0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type=str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=512)
ap.add_argument('--checkpoint', dest='checkpoint', action='store', default='/checkpoint.pth')

pa = ap.parse_args()
where = pa.data_dir  # 目录
lr = pa.learning_rate  # 学习率
structure = pa.arch  # 模型
dropout = pa.dropout  # 防止过拟合的正则
hidden_layer1 = pa.hidden_units  # 隐藏层神经元数量
power = pa.gpu  # GPU
epochs = pa.epochs  # 轮次
checkpoint = pa.checkpoint # 检查点



def train_network(model, criterion, optimizer, epochs=3, print_every=20, checkpoint='', power='gpu'):
    train_dir = where + '/train'
    valid_dir = where + '/valid'
    test_dir = where + '/test'

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(30),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([.485, .456, .406],
                                                                [.229, .224, .225])
                                           ])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([.485, .456, .406],
                                                               [.229, .224, .225])
                                          ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=True)

    steps = 0
    running_loss = 0

    print("--------------Training is starting------------- ")
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            if torch.cuda.is_available() and power == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy = 0

                for ii, (inputs2, labels2) in enumerate(validloader):
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        inputs2, labels2 = inputs2.to('cuda:0'), labels2.to('cuda:0')
                        model.to('cuda:0')

                    with torch.no_grad():
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs, labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                vlost = vlost / len(validloader)
                accuracy = accuracy / len(validloader)

                print("Epoch: {}/{}... ".format(e + 1, epochs),
                      "Loss: {:.4f}".format(running_loss / print_every),
                      "Validation Lost {:.4f}".format(vlost),
                      "Accuracy: {:.4f}".format(accuracy))

                running_loss = 0

    print("-------------- Finished training -----------------------")
    print("Dear User I the ulitmate NN machine trained your model. It required")
    print("----------Epochs: {}------------------------------------".format(epochs))
    print("----------Steps: {}-----------------------------".format(steps))
    print("That's a lot of steps")

    if checkpoint:
        print('save checkpoint to:', checkpoint)
        model.class_to_idx = train_data.class_to_idx
        torch.save({'structure': structure,
                    'hidden_layer1': hidden_layer1,
                    'dropout': dropout,
                    'lr': lr,
                    'nb_of_epochs': epochs,
                    'state_dict': model.state_dict(),
                    'class_to_idx': model.class_to_idx},
                   checkpoint)


model, criterion, optimizer = futils.setup(structure, dropout, hidden_layer1, lr, power)
train_network(model, criterion, optimizer, epochs, 20, checkpoint, power)