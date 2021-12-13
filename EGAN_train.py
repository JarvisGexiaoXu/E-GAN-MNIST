import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from model import discriminator, generator
import numpy as np
import matplotlib.pyplot as plt
import random

start_time = time.time()
plt.rcParams['image.cmap'] = 'gray'

def show_images(images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index+1)
        plt.imshow(image.reshape(28, 28))


# Discriminator Loss => BCELoss
def d_loss_function(inputs, targets):
    return nn.BCELoss()(inputs, targets)

# Mutations
def g_BCE_function(inputs):
    targets = torch.ones([inputs.shape[0], 1])
    targets = targets.to(device)
    return nn.BCELoss()(inputs, targets)

def g_MAE_function(inputs):
    targets = torch.ones([inputs.shape[0], 1])
    targets = targets.to(device)
    return nn.SmoothL1Loss()(inputs, targets)

def g_MSE_function(inputs):
    targets = torch.ones([inputs.shape[0], 1])
    targets = targets.to(device)
    return nn.MSELoss()(inputs, targets)

def fitnessScore(G, D, testloader):
    num_output = 0
    fq = 0
    fd = 0
    d1 = 0
    d2 = 0
    for data in testloader:
        real_inputs = data[0].to(device)
        test = 255 * (0.5 * real_inputs[0] + 0.5)
        real_inputs = real_inputs.view(-1, 784)
        real_outputs = D(real_inputs)
        real_outputs_lst = real_outputs.tolist()

        noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)
        fake_outputs_lst = fake_outputs.tolist()
        
        for i in range(len(fake_outputs_lst)):
            for j in range(len(fake_outputs_lst[i])):
                num_output += 1
                # fq
                fq += fake_outputs_lst[i][j]
                # fd
                d1 += np.log(real_outputs_lst[i][j])
                d2 += np.log(1 - fake_outputs_lst[i][j])
        outputs = real_outputs + fake_outputs
        gradients = torch.autograd.grad(outputs, inputs=D.parameters(),
                                        grad_outputs=torch.ones(outputs.size()).to(device),
                                        create_graph=True,retain_graph=True,only_inputs=True)         
    gradients = list(gradients)
    with torch.no_grad():
        for i, grad in enumerate(gradients):
            grad = grad.view(-1)
            allgrad = grad if i == 0 else torch.cat([allgrad,grad]) 
    fd = -torch.log(torch.norm(allgrad)).data.cpu().numpy()
        
    fq = fq/num_output
    return fq, fd

def initialization(populationNum):
    G = []
    g_optimizer = []
    for i in range(populationNum):
        G.append(generator().to(device))
        g_optimizer.append(optim.Adam(G[i].parameters(), lr=lr, betas=(0.5, 0.999)))        
    return G, g_optimizer
'''
# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)
'''

# Settings
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
epochs = 200
lr = 0.0002
batch_size = 32
population_size = 3
path = "parent.pth"

# Model
G, g_optimizer = initialization(population_size)
D = discriminator().to(device)
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
print(G)
print(D)
# Transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])


# Load data
train_set = datasets.MNIST('mnist/', train=True, download=True, transform=transform)
test_set = datasets.MNIST('mnist/', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# compute fitness
fitness_Lst = []
for i in range(len(G)):
    fq,fd = fitnessScore(G[i],D,test_loader)
    fitness_Lst.append(fq + 0.1 * fd)
print(fitness_Lst)

for epoch in range(epochs):
    epoch += 1
    for times, data in enumerate(train_loader):
        # parent selection
        print("1-parent selection")
        index = random.randint(0, 2)
        times += 1
        real_inputs = data[0].to(device)
        test = 255 * (0.5 * real_inputs[0] + 0.5)

        real_inputs = real_inputs.view(-1, 784)
        real_outputs = D(real_inputs)
        real_label = torch.ones(real_inputs.shape[0], 1).to(device)

        noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        fake_inputs = G[index](noise)
        fake_outputs = D(fake_inputs)
        fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

        outputs = torch.cat((real_outputs, fake_outputs), 0)
        targets = torch.cat((real_label, fake_label), 0)
        # Zero the parameter gradients
        d_optimizer.zero_grad()


        # Backward propagation
        d_loss = d_loss_function(outputs, targets)
        if(d_loss.item()>0.5):
            print(d_loss.item())
            d_loss.backward()
            d_optimizer.step()
        # Generator
        noise = (torch.rand(real_inputs.shape[0], 128)-0.5)/0.5
        noise = noise.to(device)

        fake_inputs = G[index](noise)
        fake_outputs = D(fake_inputs)
        
        # Implement three different mutations
        offspring = []
        offspring_optimizer = []
        offspring_fitness = []
        g_loss = []
        torch.save(G[index].state_dict(), path)
        for i in range(3):
            child = generator().to(device)
            child.load_state_dict(torch.load(path))
            child.eval()
            offspring.append(child)
            offspring_optimizer.append(optim.Adam(child.parameters(), lr=lr, betas=(0.5, 0.999)))
            print("2-mutation...")
            if i == 0:
                g_loss.append(g_BCE_function(fake_outputs))
            elif i == 1:
                g_loss.append(g_MAE_function(fake_outputs))
            else:
                g_loss.append(g_MSE_function(fake_outputs))
            offspring_optimizer[i].zero_grad()
            g_loss[i].backward(retain_graph=True)
            offspring_optimizer[i].step()
            # Compute offspring fitness
            fd,fq = fitnessScore(offspring[i],D,test_loader)
            offspring_fitness.append(fd + 0.1 * fq)
        print("3-survivor selection")
        # Survivor Selection
        G = G + offspring
        g_optimizer = g_optimizer + offspring_optimizer
        fitness_Lst = []
        for i in range(len(G)):
            fd,fq = fitnessScore(G[i],D,test_loader)
            fitness_Lst.append(fd + 0.1 * fq)
        
        for i in range(population_size):
            x = fitness_Lst.index(min(fitness_Lst))
            del G[x]
            del g_optimizer[x]
            del fitness_Lst[x]
        print(fitness_Lst)
        x = fitness_Lst.index(max(fitness_Lst))
        fq,fd = fitnessScore(G[x],D,test_loader)
        if times % 2 == 0 or times == len(train_loader):
            g = g_BCE_function(fake_outputs)
            print('[{}/{}, {}/{}] D_loss: {:.3f} G_loss: {:.3f}'.format(epoch, epochs, times, len(train_loader), d_loss.item(), g.item()))
            print("highest fitness: ", max(fitness_Lst))
    if epoch % 10 == 0:
        imgs_numpy = (fake_inputs.data.cpu().numpy()+1.0)/2.0
        show_images(imgs_numpy[:16])
        plt.show()      
print('Training Finished.')
print('Cost Time: {}s'.format(time.time()-start_time))                



