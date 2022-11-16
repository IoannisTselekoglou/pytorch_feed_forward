import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



#hpyer parameters


input_size = 748 # 28x28 images 

hidden_size = 100 

num_classes = 10 # digits from 0 -> 10 

num_epochs = 2

batch_size = 100

learning_rate = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#MNIST import

train_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                           transform=transforms.ToTensor(), download=False)
test_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                           transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                           shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples)



#for i in range(6):
#    plt.subplot(2,3,i+1)
#    plt.imshow(samples[i][0], cmap="gray")
##plt.show()


#Basic neuralnet , first init layers than forward function

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) #Linear function
        self.relu = nn.ReLU() #Activation Function
        self. l2 = nn.Linear(hidden_size, num_classes) #Lienar function


    def forward(self, x): #function which binds layers togeher
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

#create model
model = NeuralNetwork(input_size, hidden_size, num_classes)

#next step Lossfunction and optimizer

crit = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #100 , 1 , 28, 28 shape
        #input = 784 -> 100, 784 so reshape
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        #forward pass
        outputs = model(images)
        loss = crit(outputs, labels)
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (i+1) % 100 == 0:
            print(f"epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}")





#test

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        #value , index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(acc)




