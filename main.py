import json, argparse
import torch.optim as optim
import torch as torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from model import Net
from dataset import DataSet

# Function finds the number of correct predictions given a list of training and test data
def test(outputs, labels):

    # number of images we correctly predicted
    correct = 0

    for output, label in zip(outputs, labels):

        # Pick the indice with maximal probability
        if torch.argmax(output) == torch.argmax(label):
            correct += 1

    # Calc Average
    correct = correct/len(outputs) * 100

    return correct

if __name__=="__main__":

    # Hyperparameters from json file
    with open("params.json") as paramfile:
        hyper = json.load(paramfile)

    # Create the transformations
    transform = transforms.Compose([transforms.CenterCrop(200), transforms.Grayscale(),transforms.ToTensor()])

    # Load in a test and train set
    train_set = DataSet("traindata", transform=transform)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=1)
    test_set = DataSet("testdata", transform=transform)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1)

    # Create our network and dataset
    model = Net()

    # Define optimizer and loss function
    optimizer = optim.SGD(model.parameters(),lr=hyper["learning rate"])
    loss_func = torch.nn.BCELoss(reduction="mean")

    # Number of training epochs
    num_epochs = hyper["num epochs"]

    # lists to track preformance of network
    obj_vals= []
    cross_vals= []

    # Training loop
    for epoch in range(1, num_epochs + 1):
        
        # Track the outputs/labels of the model every epoch
        outputs = []
        labels = []

        for inputs, label in trainloader:
    
            # Clear our gradient buffer
            optimizer.zero_grad()

            # Clear gradients
            model.zero_grad()

            # feed our inputs through the net
            output = model(inputs)

            # Append list to calculate % correct this epoch
            outputs.append(output)
            labels.append(label)

            # Calculate our loss
            loss = loss_func(output, torch.flatten(label))

            # Backpropagate our loss
            loss.backward()

            optimizer.step()

        # track our progress
        obj_vals.append(loss)
        test_val = model.test(testloader, loss_func, epoch)
        cross_vals.append(test_val)

        # High verbosity report in output stream
        if hyper["v"]>=2:
            if not ((epoch + 1) % hyper["display epochs"]):
                print('Epoch [{}/{}]'.format(epoch, num_epochs) +\
                    '\tTraining Loss: {:.4f}'.format(loss) +\
                    '\tTest Loss: {:.4f}'.format(test_val) +\
                    "\tPercent Correct: {:.2f}".format(test(outputs, labels)))

    # Low verbosity final report
    if hyper["v"]:
        print('Final training loss: {:.4f}'.format(obj_vals[-1]))
        print('Final test loss: {:.4f}'.format(cross_vals[-1]))
        print("Final percent correct: {:.2f}".format(test(outputs, labels)))

    # Save model
    torch.save(model.state_dict(), "model")