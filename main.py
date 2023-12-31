#Importing necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision
from torchvision import datasets
import torchvision.transforms.v2 as v2

#Making the code device agnostic
device='cuda' if torch.cuda.is_available() else 'cpu'

#Performing data augumentation to increase accuracy
transformTrain=v2.Compose([v2.RandomResizedCrop(size=(32,32), scale=(0.8,1.0), antialias=True),
                      v2.RandomHorizontalFlip(p=0.5),
                      #v2.RandomVerticalFlip(p=0.5),
                      v2.PILToTensor(),
                      v2.ToDtype(torch.float32, scale=True),
                    ])

transformTest=v2.Compose([v2.PILToTensor(),
                          v2.ToDtype(torch.float32, scale=True),
                        ])

#Loading training and testing data
trainData=datasets.CIFAR10(root='data',
                           train=True,
                           download=True,
                           transform=transformTrain
                           )

testData=datasets.CIFAR10(root='data',
                           train=False,
                           download=True,
                           transform=transformTest
                           )

#Setting batch size
batchSize=16

#Turning the loaded data into iterable batches
trainDataLoader=DataLoader(trainData,
                           batch_size=batchSize,
                           shuffle=True
                           )

testDataLoader=DataLoader(testData,
                           batch_size=batchSize,
                           shuffle=True
                           )

#Defining the model
model=nn.Sequential(
                    #Simple Convolution to reduce number of inputs and help recognise patterns
                    nn.Conv2d(3,16,3,padding='same'),
                    nn.ReLU(),
                    nn.Conv2d(16,32,3,padding='same'),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2),

                    nn.Conv2d(32,64,3,padding='same'),
                    nn.ReLU(),
                    nn.Conv2d(64,64,3,padding='same'),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2),

                    nn.Conv2d(64,128,3,padding='same'),
                    nn.ReLU(),
                    nn.Conv2d(128,128,3,padding='same'),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2),

                    nn.Flatten(),

                    #Using dropout to reduce dependancy on certainnneurons and thus overfitting
                    nn.Dropout(0.5),

                    #Deeper networks are able to learn patterns to greater accuracy than their shallow counterparts
                    nn.Linear(4*4*128,512),
                    nn.Linear(512,256),
                    nn.Linear(256,256),
                    nn.Linear(256,128),
                    nn.Linear(128,128),
                    nn.Linear(128,10),
                    )

#Setting up loss function, optimizer, and scheduler
lossFn=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.0001)

#Model tends to converge around 60 to 65 epochs
epochs=65

#Iterating through the training and test dataloaders
for i in range(epochs):
    #Shifting the model to the GPU if applicable
    model.to(device)
    trainLoss=0
    trainAcc=0
    model.train()
    #Training the model
    for j,data in enumerate(trainDataLoader):
        #Obtaining the image and label from the dataset
        img,label=data

        #Moving image, label to GPU if applicable 
        img=img.to(device)
        label=label.to(device)
        
        #Setting the gradient to 0
        optimizer.zero_grad()

        #Using the model to make a prediction
        out=model(img)
        
        #Calculating Accuracy of the output of the model
        correct = torch.eq(label, out.argmax(dim=1)).sum().item()
        trainAcc += (correct / len(out)) * 100

        #Calculating Loss and Gradient
        loss=lossFn(out,label)
        trainLoss+=loss.item()
        loss.backward()
        
        #Adjusting the weights
        optimizer.step()

    

    testLoss=0
    testAcc=0
    model.eval()

    #Evaluatig the model
    with torch.no_grad():
        for j,data in enumerate(testDataLoader):
            #Obtaining the image and label from the dataset
            img,label=data

            #Moving image, label to GPU if applicable 
            img=img.to(device)
            label=label.to(device)
            
            #Using the model to make a prediction
            out=model(img)
            
            #Using the model to make a prediction
            correct = torch.eq(label, out.argmax(dim=1)).sum().item()
            testAcc +=(correct / len(out)) * 100
            
            #Calculating Loss
            loss=lossFn(out,label)
            testLoss+=loss.item()

    #Rounding off the various calculated loss and accuracy values to enhance readability and calculating their average for the batch
    trainLoss=round(trainLoss/len(trainDataLoader),5)
    testLoss=round(testLoss/len(testDataLoader),5)

    trainAcc=round(trainAcc/len(trainDataLoader),2)
    testAcc=round(testAcc/len(testDataLoader),2)

    #Printing the various calculated loss and accuracy values to evaluate model performance
    print(f'Epoch: {i}')
    print(f'Train Loss: {trainLoss} | Train Accuracy: {trainAcc}%')
    print(f'Test Loss: {testLoss} | Test Accuracy: {trainAcc}%')

#Saving the model
torch.save(model.state_dict(),'model')
