import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as utils
import random
import os

NUM_EPOCH = 10



class ResNet50_CIFAR(nn.Module):
    def __init__(self):
        super(ResNet50_CIFAR, self).__init__()
        # Initialize ResNet 50 with ImageNet weights
        ResNet50 = models.resnet50(pretrained=True)
        modules = list(ResNet50.children())[:-1]
        backbone = nn.Sequential(*modules)
        # Create new layers
        self.backbone = nn.Sequential(*modules)
        self.fc1 = nn.Linear(2048, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, img):
        # Get the flattened vector from the backbone of resnet50
        out = self.backbone(img)
        # processing the vector with the added new layers
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Define the training dataloader
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10('./data', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
    ##

    ## Create model, objective function and optimizer
    model = ResNet50_CIFAR()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.fc1.parameters()) + list(model.fc2.parameters()),
                           lr=0.001, momentum=0.9)

    ## Do the training
    for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
    print('Finished Training')

    # save the model
    torch.save(model.state_dict(), 'model.pt')

# based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

def test():
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # load model
    model = ResNet50_CIFAR()
    model.load_state_dict(torch.load('model.pt', map_location='cpu'))
    model = model.cpu()

    results_list = []
    images_list = []
    outputs_list = []
    labels_list = []
    predicted_list = []
    i = 0
    max_iter = 100
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            images_list += list(images)
            labels_list += list(labels)
            outputs_list += list(outputs)
            predicted_list += list(predicted)
            i = i+1
            if i == max_iter:
                break

    for i in range(len(images_list)):
        image = images_list[i]
        label = labels_list[i]
        output = outputs_list[i]
        predicted = predicted_list[i]
        img_path = 'images/' + str(i + 1) + '.png'
        torchvision.utils.save_image(image, img_path)
        scores = softmax(output.numpy())
        results_list += [(img_path, classes[predicted], classes[label], scores)]


    with open('HTML.html', 'w') as file:
        file.write("""
        <html>
        	<head>
                <title>Tests</title>
                <style>
                table {
                    border-collapse: collapse;
                    width: 100%;
                    text-align: center;
                }
                th {
                    color: black;
                }
                </style>
        	</head>
        	<body>
        	<table border='1'>
            <thead>
        		<tr>
        			<th width='64px'>image</th>
        			<th width='64px'>prediction</th>
        			<th width='64px'>plane</th>
        			<th width='64px'>car</th>
        			<th width='64px'>bird</th>
        			<th width='64px'>cat</th>
        			<th width='64px'>deer</th>
        			<th width='64px'>dog</th>
        			<th width='64px'>frog</th>
        			<th width='64px'>horse</th>
        			<th width='64px'>ship</th>
        			<th width='64px'>truck</th>
        		</tr>
            </thead>
        """)

        for (img_path, predicted_label, true_label, scores) in results_list:
            file.write("<tr>\n")
            file.write("<td width='100px'>\n<img src=\"%s\" height=100 width=100>\n</td>\n" % (img_path))
            file.write("<td width='100px'>\n<h2>%s</h2>\n</td>\n" % (predicted_label))
            for score in scores:
                if predicted_label != true_label:
                    file.write("<td width='100px' style='color:%s'>\n%.8f\n</td>\n" % ('red', score))
                else:
                    file.write("<td width='100px' style='color:%s'>\n%.8f\n</td>\n" % ('green', score))
            file.write("</tr>\n")

        file.write("</table></body>\n</html>")



if __name__ == '__main__':

    #train()
    test()

