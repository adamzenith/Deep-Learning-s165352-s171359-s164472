import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
#tqdm = partial(tqdm, position=0, leave=True)
from torch.nn.parameter import Parameter
#from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
import wandb
import plotly
from scipy.io import loadmat
import pandas as pd
from torch import tanh,sigmoid
#from torch.nn.functional import leaky_relu as relu, tanh
from torch.optim.lr_scheduler import StepLR
import plotly.graph_objects as go
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
from rich import print

from torch.nn.functional import leaky_relu as relu, tanh


#Hyperparameters
#----------------------------------------------------------------------------------------------------------------------------
# Here is where the hyper parameters are tuned, every run is a sweep in wandb because it was convenient at the time.
#
#
#
#
#----------------------------------------------------------------------------------------------------------------------------   
""""sweep_config = {
    "name" : "Burger Sweep",
    "method" : "bayes",
    "metric" : {
        "name" : "Validation_loss",
        "goal" : "minimize"
    },
    "parameters" : {
        "epochs": { "min" : 50,"max" : 500},
        "batch_size" : {"min" : 1, "max" : 20},
        "gridRes" : {"min" : 100,"max" : 10000},
        "nodes_per_layer" : {"value" : 10},
        "lambda1" : {"min" : 1.0,"max" : 10.0},
        "lambda2" : {"min" : 1.0,"max" : 10.0},
        "lambda3" : {"min" : 1.0,"max" : 10.0},
        "lambda4" : {"min" : 1.0,"max" : 10.0},
        #"BCsample" : {"min" : 10,"max" : 500},
        #"ICsample" : {"min" : 10,"max" : 500},
        "val_num" : {"value" : 1},
        "sample" : {"value" : 100},
        "epsilon" : {"value" : 0.01/math.pi},
        "learning_rate" : {"value" : 0.01},
        #"decrease" : {"min" : 1 ,"max" : 20}
    }
}"""

sweep_config = {
    "name" : "Burger Sweep",
    "method" : "random",
    "metric" : {
        "name" : "Validation_loss",
        "goal" : "minimize"
    },
    "parameters" : {
        "epochs": { "value" : 300},
        "batch_size" : {"value": 10},
        "gridRes" : {"value" :9700},
        "nodes_per_layer" : {"value" : 10},
        "lambda1" : {"min" : 1,"max" : 5},
        "lambda2" : {"min" : 1,"max" : 5},
        "lambda3" : {"min" : 1,"max" : 5},
        "lambda4" : {"min" : 1,"max" : 5},
        #"BCsample" : {"min" : 10,"max" : 500},
        #"ICsample" : {"min" : 10,"max" : 500},
        "val_num" : {"value" : 1},
        "sample" : {"value" : 100},
        "epsilon" : {"value" : 0.01/math.pi},
        "learning_rate" : {"value" : 0.01},
        "activation" : {"values" : ["tanh","sigmoid"]}
         
        #"decrease" : {"min" : 1 ,"max" : 20}
    }
}
#Model
#----------------------------------------------------------------------------------------------------------------------------
#Here the FFNN is defined, not the one with modular layers but we could easily swap that out
#
#
#
#
#----------------------------------------------------------------------------------------------------------------------------   



#Training Loop
#----------------------------------------------------------------------------------------------------------------------------
# The training loop is defined in a function so it can be called by wandb
#
#
#
#
#----------------------------------------------------------------------------------------------------------------------------   
def train(config=sweep_config):


  with wandb.init(config=config):
    config=wandb.config
    activation=config.activation
    class Net(nn.Module):
      def __init__(self,dim,L1):
          super(Net, self).__init__()
          self.drop = nn.Dropout(0.1)
          self.l_1 = nn.Linear(in_features=dim,
                              out_features=L1,
                              bias=True)
          self.l_2 = nn.Linear(in_features=L1,
                              out_features=L1,
                              bias=True)
          self.l_3 = nn.Linear(in_features=L1,
                              out_features=L1,
                              bias=True)
          self.l_4 = nn.Linear(in_features=L1,
                              out_features=L1,
                              bias=True)
          self.l_5 = nn.Linear(in_features=L1,
                              out_features=L1,
                              bias=True)
          self.l_out = nn.Linear(in_features=L1,
                              out_features=1,
                              bias=True) #IF EVERYTHING  BROKEN CHANGE THIS BACK TO FALSE
          for param in self.parameters():
            if len(param.shape) > 1:
              nn.init.xavier_normal_(param)

      if activation == "tanh":  
        def forward(self,x):
          
            x = tanh(self.l_1(x))
            #x = self.drop(x)
            x = tanh(self.l_2(x))
            #x = self.drop(x)
            x = tanh(self.l_3(x))
            #x = self.drop(x)
            x = tanh(self.l_4(x))
            #x = self.drop(x)
            x = tanh(self.l_5(x))
            #x = self.drop(x)
            x = (self.l_out(x))
            
            return x
      elif activation == "sigmoid":
        def forward(self,x):
          
            x = sigmoid(self.l_1(x))
            #x = self.drop(x)
            x = sigmoid(self.l_2(x))
            #x = self.drop(x)
            x = sigmoid(self.l_3(x))
            #x = self.drop(x)
            x = sigmoid(self.l_4(x))
            #x = self.drop(x)
            x = sigmoid(self.l_5(x))
            #x = self.drop(x)
            x = (self.l_out(x))
            
            return x
    #Data Loading
    data = loadmat('burgers_shock.mat')
    x = data['x'].squeeze()
    t = data['t'].squeeze()
    u_val = data['usol']
    X,T=np.meshgrid(x,t)
    positions = np.vstack([T.ravel(), X.ravel()])


    

    #Extracting hyperparameters from the dictionary defined above
    
    epsilon=config.epsilon #Viscosity term
    epochs = config.epochs # Number of epoch
    batch_size = int(config.gridRes/config.batch_size) # full batches 

    nodes_per_layer = config.nodes_per_layer # units per layer

    lambda1 = config.lambda1 # Weight for the PDE loss
    lambda2 = config.lambda2 # Weight for the BC loss
    lambda3 = config.lambda3 # Weight for the IC loss
    lambda4 = config.lambda4 # Weight for the exact sol

    gridRes = config.gridRes # Number of datapoints within the bounds
    BCSample = config.sample # Number of BC samples
    ICSample = config.sample # Number of IC samples
    val_num = config.val_num # Number of validations
    
    learning_rate = config.learning_rate # Learning rate for Adam


    #Here the data is defined, generated uniformly randomly in the range we are in.
    n=gridRes
    datapoints=n
    input = np.random.uniform(low=[0,-1],high=[1,1],size=(n,2))
    input = torch.Tensor(input)
    input.requires_grad=True

    #Initial boundary: ic is x = -1 .. 1, zeros = array of just 0
    ic = np.linspace(-1,1,ICSample) 
    zeros=ic*0

    # Boundary: bc is t = 0..1 , ones array of just 1 and mones is array of -1
    bc = np.linspace(0,1,BCSample)
    zeros_bc=bc*0
    ones = zeros_bc+1 #Set the zero array to a array of just 1
    mones = zeros_bc-1#Set the zero array to a array of just -1
    

    #Converted to tensor
    data = TensorDataset(input) #To wrap the tensor data, Each sample will be retrieved by indexing tensors along the first dimension.
    loader=DataLoader(data,batch_size = batch_size,shuffle=True,drop_last=True) # Shuffle all data and seperate into batches

    #Initializing model
    burgerNet=Net(2,nodes_per_layer)

    # Define optimizers
    optimizer = optim.Adam(burgerNet.parameters(),lr=learning_rate)
    steplr = StepLR(optimizer, 
                step_size=2000, 
                gamma=0.7) # After 2000 epochs the learning rate of the optimizer is multiplied by gamma. So the value for the learning rate is declining across training!

    #Training Loop
    for epoch in tqdm(range(epochs),leave=False, position=0):  #Burgers equation in the box t=[0,1], x=[-1,1]

      for batch in loader:
        
        optimizer.zero_grad() #Reset the gradients
        batch = batch[0] #this is necessary for dataloader


        u = burgerNet(batch) # Forward

        
        # Calculate the derivatives of the differential equation
        grads = torch.autograd.grad(u,batch,torch.ones_like(u),create_graph=True)[0]
        u_t = grads[:,0] #Extract the partial derivative in respect to t
        u_x = grads[:,1] #Extract the partial derivative in respect to x
        uu_x = u[:,0]*u_x #Calculate other part for the differential eq.

        #Calculate the double derivative with respect to x_
        #u_xx = torch.autograd.grad(u_x,batch,torch.ones_like(u_x),create_graph=True,allow_unused=True)[0][:,1]
        u_xx = torch.autograd.grad(grads,batch,torch.ones_like(grads),create_graph=True)[0][:,1]

        #print(train_positions.shape,u_train.shape)

        # Construct the loss terms:
        step = int(25600/val_num)
        #Validation loss, exact solution - models solution
        Val_loss = torch.sum((burgerNet(torch.Tensor(positions[:,0::step].T))[:,0]-torch.Tensor(u_val.T.ravel()[0::step]))**2)/len(u_val.ravel())
        
        # Differential equation solution PDE
        PDEloss = torch.sum((u_t+uu_x-epsilon*u_xx)**2)/datapoints #Burger Equation loss  

        # Initial condition loss
        ICloss = ((burgerNet((torch.Tensor(np.transpose([zeros,ic]))))[:,0]-(-(torch.sin(torch.Tensor([math.pi*ic]))[0])))**2)
        ICloss = torch.sum(ICloss)/ICSample

        # Boundary condition loss
        BCloss = burgerNet((torch.Tensor(np.transpose([bc,mones]))))**2 + burgerNet((torch.Tensor(np.transpose([bc,ones]))))**2
        BCloss = torch.sum(BCloss)/(BCSample*2)

        # Weighted sum of all the losses
        Loss = lambda1*PDEloss+lambda2*BCloss+lambda3*ICloss#+Val_loss*lambda4

        # Data logging in WandB
        wandb.log({"PDEloss" : PDEloss})
        wandb.log({"ICloss" : ICloss})
        wandb.log({"BCloss" : BCloss})
        wandb.log({"Loss" : Loss, "epoch" : epoch})
        wandb.log({"vallos" : Val_loss})


        # Backward pass:
        Loss.backward()
        optimizer.step()
        steplr.step()
        

    # Run trained model and visualize it:
    model=burgerNet
    gridRes=200
    input = np.meshgrid(np.arange(gridRes+1)/gridRes,np.arange(-gridRes,gridRes+1)/(gridRes))

    datapoints=(gridRes*2+1)*(gridRes+1)
    input = np.reshape(input,(2,datapoints))
    input = np.transpose(input)
    input = torch.Tensor(input)
    x = np.transpose(input.detach().numpy()) #For the plot

    output = burgerNet(input).detach().numpy() #forward åass

    fig1 = plt.figure(dpi=150)
    ax= plt.scatter(x[0],x[1],c=output,cmap="coolwarm")
    fig1.colorbar(ax)
    wandb.log({"colorbar" : wandb.Image(fig1)})
    
    # Load exact solution to visualize the results
    data = loadmat('burgers_shock.mat')
    x = data['x'].squeeze()
    t = data['t'].squeeze()
    u = data['usol']
    actual_outputs_1 = pd.read_excel (r't=0.25.xlsx',engine="openpyxl")
    actual_outputs_2 = pd.read_excel (r't=0.50.xlsx',engine="openpyxl")
    actual_outputs_3 = pd.read_excel (r't=0.75.xlsx',engine="openpyxl")
    

    x_t_25=np.hstack(( X[0:1,:].T, T[25:26,:].T)) # Data at t = 0.25
    x_t_50=np.hstack(( X[0:1,:].T, T[50:51,:].T))# Data at t = 0.5
    x_t_75=np.hstack(( X[0:1,:].T, T[75:76,:].T))# Data at t = 0.75
    x_t_25[:, [0,1]] = x_t_25[:, [1,0]]
    x_t_50[:, [0,1]] = x_t_50[:, [1,0]]
    x_t_75[:, [0,1]] = x_t_75[:, [1,0]]
    fig, axs = plt.subplots(1, 3, figsize=(8,4) ,sharey=True,dpi=150)
    l1,=axs[0].plot(actual_outputs_1['x'], actual_outputs_1['t'],linewidth=6,color='b')
    l2,=axs[0].plot(x_t_25[:,1],model(torch.Tensor(x_t_25)).detach().numpy(),linewidth=6,linestyle='dashed',color='r')
    axs[0].set_title('t=0.25')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('u (x,t)')


    axs[1].plot(actual_outputs_2['x'],actual_outputs_2['t'],linewidth=6,color='b')
    axs[1].plot(x_t_50[:,1],model(torch.Tensor(x_t_50)).detach().numpy(),linewidth=6,linestyle='dashed',color='r')
    axs[1].set_title('t=0.50')
    axs[1].set_xlabel('x')

    axs[2].plot(actual_outputs_3['x'], actual_outputs_3['t'],linewidth=6,color='b')
    axs[2].plot(x_t_75[:,1],model(torch.Tensor(x_t_75)).detach().numpy(),linewidth=6,linestyle='dashed',color='r')
    axs[2].set_title('t=0.75')
    axs[2].set_xlabel('x')

    #line_labels = ['Exact','Predicted']

    fig.legend(handles=(l1,l2),labels=('Exact','Predicted'),loc='upper right')
    wandb.log({"sins: " : wandb.Image(fig)})
    fig.show()

    Validation_loss = torch.sum((burgerNet(torch.Tensor(positions.T))[:,0]-torch.Tensor(u.T.ravel()))**2)/len(u.ravel())
    
    wandb.log({"Validation_loss" : Validation_loss})
    
#----------------------------------------------------------------------------------------------------------------------------
#
#
#
#
#
#----------------------------------------------------------------------------------------------------------------------------   
      

sweep_id = wandb.sweep(sweep_config,project = "tanh vs sigmoid")

wandb.agent(sweep_id, train,count = 30)