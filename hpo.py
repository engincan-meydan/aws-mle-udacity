#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import copy
import argparse
import os
import logging
import sys
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger=logging.getLogger(__name__) #means of tracking events
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):
    
    model.eval() 
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        
        inputs = inputs.to(device) #GPU
        labels = labels.to(device) #GPU
        
        outputs=model(inputs) #first we get predictions
        loss=criterion(outputs, labels)
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss // len(test_loader) #loss
    total_acc = running_corrects.double() // len(test_loader) #accuracy
    
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")
    
#Report events that occur during normal operation of a program
    
#     print ('Printing Log')
#     print(
#         "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
#             total_loss, running_corrects, len(test_loader.dataset), 100.0 * acc
#         )
#     )
    
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''

def train(model, train_loader, validation_loader, criterion, optimizer, device):
    
    epochs=5
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        for phase in ['train', 'valid']:
            if phase=='train':
                model.train()
            else:
                model.eval()
                
            # model would be in two different modes
                
            running_loss = 0.0 #loss
            running_corrects = 0 #predicted correctly

            for inputs, labels in image_dataset[phase]:
                
                inputs = inputs.to(device) #GPU
                labels = labels.to(device) #GPU
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train': #only happening with training
                    optimizer.zero_grad() #gradient reset
                    loss.backward() # backward prop
                    optimizer.step() # updating the parameters

                _, preds = torch.max(outputs, 1) #here are predictions
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss // len(image_dataset[phase])
            epoch_acc = running_corrects // len(image_dataset[phase])
            
            if phase=='valid':
                if epoch_loss<best_loss: #if current epoch loss is less, update the best
                    best_loss=epoch_loss
                else:
                    loss_counter+=1 #counter so we will stop epochs


            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase, epoch_loss, epoch_acc, best_loss))
        if loss_counter==1:
            break
        if epoch==0:
            break
            
    return model
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
def net():
    
    model = models.resnet50(pretrained=True) # I am using resnet50 like we will be using in the next exercise
    
    num_features = model.fc.in_features #got the fix thanks to the mentor's help

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(num_features, 128), #simple NN steps to keep it less complicated
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model    
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''

# def create_data_loaders(data, batch_size):
    
#     train_data_path = os.path.join(data, 'train')
#     test_data_path = os.path.join(data, 'test')
#     validation_data_path=os.path.join(data, 'valid')

#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         ])

#     test_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         ])
#     train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
#     train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

#     test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
#     test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

#     validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
#     validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    
#     return train_data_loader, test_data_loader, validation_data_loader    

def create_data_loaders(train_dir, test_dir, eval_dir, train_batch_size, test_batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''

    #transforming to prepare data loaders
    
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testing_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_data = ImageFolder(root=train_dir, transform=training_transform)
    test_data = ImageFolder(root=test_dir, transform=testing_transform)
    validation_data = ImageFolder(root=eval_dir, transform=testing_transform)

    train_loader = DataLoader(train_data, train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, test_batch_size, shuffle=False)
    validation_loader = DataLoader(validation_data, test_batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader


    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''

def main(args):
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    print(f'Log entry: Train batch size:{args.batch_size}')
    print(f'Log entry: Learning rate:{args.lr}')
    print(f'Log entry: Number of epochs:{args.epochs}')
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    
    # Here were go with GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    

    model = net()
    model = model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss() #using crossentropyloss which is used for Multiclass Classification
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr) #using adagrad which has adaptive learning rate
    
#     from next project
#.    criterion = nn.CrossEntropyLoss(ignore_index=133)
#     optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
#     train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batch_size)

    train_loader, validation_loader, test_loader = create_data_loaders(args.train_dir, args.test_dir, args.eval_dir, args.batch_size, args.test_batch_size)

        
    model=train(model, train_loader, validation_loader, criterion, optimizer, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    
    path = './hyperparameter_optimization'
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    # getting help from the exercises 
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        metavar='N',
        help='batch size for training (default: 10)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        metavar='N',
        help='learning rate (default: 0.1)'
    )

    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=10,
        metavar='N',
        help='batch size for training (default: 10)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        metavar='N',
        help='number of epochs for training (default: 5)'
    )
    
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default=os.environ["SM_MODEL_DIR"]
    )
    
    parser.add_argument(
        "--train-dir", 
        type=str, 
        default=os.environ["SM_CHANNEL_TRAIN"]
    )
    
    parser.add_argument(
        "--test-dir", 
        type=str, 
        default=os.environ["SM_CHANNEL_TEST"]
    )
    
    parser.add_argument(
        "--eval-dir", 
        type=str, 
        default=os.environ["SM_CHANNEL_VAL"]
    )
        
    args=parser.parse_args()

    print(args)
    
    main(args)