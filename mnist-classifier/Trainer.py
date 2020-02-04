import matplotlib.pyplot as plt
from IPython.display import clear_output

import torch
from torch import nn, optim
import torch.nn.functional as F

def compute_accuracy(model_output, labels):
    with torch.no_grad():
        _, labels_pred = model_output.topk(1, dim=1)
        equals = labels_pred == labels.view(*labels_pred.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        return accuracy.item()

def train_epoch(dataloader, model, criterion, optimizer, batchsize=64):
    train_loss, accuracy = 0, 0
    
    model.train()
    for data, labels in dataloader:
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        accuracy += compute_accuracy(output, labels)
    
    return train_loss/len(dataloader), accuracy/len(dataloader)


def test_epoch(dataloader, model, criterion, batchsize=64):
    test_loss, accuracy = 0, 0
    
    with torch.no_grad():
        model.eval()
        for data, labels in dataloader:
            output = model(data)
            loss = criterion(output, labels)
            
            test_loss += loss.item()
            accuracy += compute_accuracy(output, labels)
    return test_loss/len(dataloader), accuracy/len(dataloader)


def plot_history(train_history, val_history, title='loss'):
    plt.figure()
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train')
    plt.plot(val_history, label='val')

    plt.xlabel('train steps')
    
    plt.legend(loc='upper right')
    plt.grid()

    plt.show()


def train(trainloader, testloader, model, criterion, optimizer, n_epochs=5, batchsize=64):
    train_loss_log, train_accuracy_log = [], []
    test_loss_log, test_accuracy_log = [], []
    
    for epoch in range(n_epochs):
        train_loss, train_accuracy = train_epoch(trainloader, model, criterion, optimizer)
        test_loss, test_accuracy = test_epoch(testloader, model, criterion)
        
        train_loss_log.append(train_loss)
        train_accuracy_log.append(train_accuracy)
        test_loss_log.append(test_loss)
        test_accuracy_log.append(test_accuracy)
        
        clear_output()
        plot_history(train_loss_log, test_loss_log)    
        plot_history(train_accuracy_log, test_accuracy_log, title='accuracy')
        print("Epoch {} validation error = {:.2%}".format(epoch + 1, 1 - test_accuracy))
        

def view_classify(img, probs):
    ''' 
    Function for viewing an image and it's predicted classes.
    '''
    probs = probs.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
