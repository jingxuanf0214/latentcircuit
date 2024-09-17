import torch
import torch.nn as nn
from connectivity import *
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
#from scipy.sparse import random
from scipy import stats
from numpy import linalg


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class Net(torch.nn.Module):
    def __init__(self, n, alpha = .2, sigma_rec=0.15, input_size=6, output_size=2,dale=False,activation = torch.nn.ReLU() ):
        super(Net, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.sigma_rec = torch.tensor(sigma_rec)
        self.n = n
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.dale = dale
        
        # Connectivity
        self.recurrent_layer = nn.Linear(self.n, self.n, bias=False)
        #self.recurrent_layer.weight.data.normal_(mean=0., std=0.25).to(device=device)
        #self.recurrent_layer.bias.data.normal_(mean=0.2, std=0).to(device=device)
        #self.recurrent_layer.bias.requires_grad = False

        self.input_layer = nn.Linear(self.input_size, self.n, bias=False)
        self.input_layer.weight.data.normal_(mean=.1, std=.01).to(device=device)

        self.output_layer = nn.Linear(self.n, self.output_size, bias=False)
        #self.output_layer.weight.data.normal_(mean=.2, std=0.01).to(device=device)

        if self.dale:
            self.recurrent_layer.weight.data,self.input_layer.weight.data,self.output_layer.weight.data,self.dale_mask, self.output_mask, self.input_mask = init_connectivity(self.n,self.input_size,self.output_size,device='cpu',radius=1.5)
        self.connectivity_constraints()


    # Dynamics
    def forward(self, u):
        t = u.shape[1]
        states = torch.zeros(u.shape[0], 1, self.n, device=device)
        batch_size = states.shape[0]

        noise = torch.sqrt(2 * self.alpha * self.sigma_rec ** 2) * torch.empty(batch_size, t, self.n).normal_(mean=0,
                                                                                                              std=1).to(
            device=device)

        for i in range(t - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * (
                     self.activation(
                self.recurrent_layer(states[:, i, :]) + self.input_layer(u[:, i, :]) + noise[:, i, :]))
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)

        return states

    def connectivity_constraints(self):
        self.input_layer.weight.data = torch.relu(self.input_layer.weight.data)
        self.output_layer.weight.data =  torch.relu(self.output_layer.weight.data)

        if self.dale:
            self.input_layer.weight.data = self.input_mask * torch.relu(self.input_layer.weight.data)

            self.output_layer.weight.data = self.output_mask * torch.relu(self.output_layer.weight.data)

            self.recurrent_layer.weight.data = torch.relu(
                self.recurrent_layer.weight.data * self.dale_mask) * self.dale_mask



    def l2_ortho(self):
        b = torch.cat((self.input_layer.weight, self.output_layer.weight.t()), dim=1)
        b = b / torch.norm(b, dim=0)
        return torch.norm(b.t() @ b - torch.diag(torch.diag(b.t() @ b)), p=2)

    def loss_function(self, x, z, mask):
        return self.mse_z(x,z,mask) + self.l2_ortho() + 0.05 * torch.mean(x**2)

    def mse_z(self, x, z, mask):
        mse = nn.MSELoss()
        return mse(self.output_layer(x)*mask, z*mask)

    def fit(self, u, z, mask, epochs = 10000, lr=.01, verbose = False, weight_decay = 0):


        my_dataset = TensorDataset(u, z,mask)  # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size=128)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay = weight_decay)
        epoch = 0
        while epoch < epochs:
            for batch_idx, (u_batch, z_batch,mask_batch) in enumerate(my_dataloader):
                optimizer.zero_grad()
                x_batch = self.forward(u_batch)
                loss = self.loss_function(x_batch, z_batch, mask_batch)
                loss.backward()
                optimizer.step()
                self.connectivity_constraints()
            epoch += 1
            if verbose:
                if epoch % 5 == 0:
                    x = self.forward(u)
                    print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
                    print("mse_z: {:.4f}".format(self.mse_z(x, z, mask).item()))

    def evaluate(self, u, z, batch_size=128):
        """
        Evaluate the accuracy of the network on a dataset by comparing the network's predicted choice
        at the last time step for each trial with the true labels, in a batch-wise manner.
        
        :param u: Input data (batch_size, time_steps, input_size)
        :param z: True labels (batch_size, time_steps, output_size), with the correct choice at the last time step
        :param batch_size: Size of the batch for evaluation
        
        :return: Accuracy (percentage of correct predictions)
        """
        # Create a DataLoader to iterate through the dataset in batches
        dataset = TensorDataset(u, z)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for u_batch, z_batch in dataloader:
                # Move data to the correct device (GPU or CPU)
                u_batch = u_batch.to(device)
                z_batch = z_batch.to(device)
                
                # Forward pass to get the network output for the batch
                x_batch = self.forward(u_batch)
                output_batch = self.output_layer(x_batch)

                # Get the network's predicted choice at the last time step
                predictions = torch.argmax(output_batch[:, -1, :], dim=1)

                # True labels for the last time step (assuming binary classification: right or left)
                true_labels = torch.argmax(z_batch[:, -1, :], dim=1)

                # Calculate the number of correct predictions
                correct_predictions = (predictions == true_labels).sum().item()
                total_correct_predictions += correct_predictions
                total_predictions += true_labels.shape[0]

        # Calculate accuracy as the percentage of correct predictions
        accuracy = total_correct_predictions / total_predictions * 100
        
        return accuracy


                

