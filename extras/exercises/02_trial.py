import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy
from torch import nn


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float) # features as float32
y = torch.from_numpy(y).type(torch.LongTensor) # labels need to be of type long

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create train and test splits
X_train, X_test, y_train ,y_test = train_test_split(X, y, shuffle=True)

acc_fn = Accuracy(task="multiclass", num_classes=3).to(device)

# Create model by subclassing nn.Module
class multi_class_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=3)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.layer_1(x)
        x = self.activation(x)
        x = self.layer_2(x)
        x = self.activation(x)
        x = self.layer_3(x)
        return x
    

model = multi_class_model()
model.to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)


X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)
# Build a training loop for the model
epochs = 5000
train_loss, test_loss = [], []
train_acc, test_acc = [],[]
# Loop over data
for epoch in range(epochs):

  ## Training
  model.train()
  # 1. Forward pass
  y_logits = model(X_train)

  y_pred = torch.argmax(y_logits, dim=1)

  # 2. Calculate the loss
  loss = loss_fn(y_logits, y_train)
  train_loss.append(loss.item())

  # 2b. Calculate the 
  acc_fn.reset()
  acc = acc_fn(y_pred, y_train)
  train_acc.append(acc.item())
  
  # 3. Optimizer zero 
  optimizer.zero_grad()

  # 4. Loss backward
  loss.backward()

  # 5. Optimizer step
  optimizer.step()
  

  ## Testing
  with torch.inference_mode():
    model.eval()
  

    # 1. Forward pass
    y_logits_test = model(X_test)
    y_pred_test = torch.argmax(y_logits_test, dim=1)
    
    # 2. Caculate loss and acc
    loss = loss_fn(y_logits_test, y_test)
    test_loss.append(loss.item())

    acc = acc_fn(y_pred_test, y_test)
    test_acc.append(acc.item())
    
  # Print out what's happening every 100 epochs
  if epoch%100 == 0:
    print(f"Epoch : {epoch} | Train_loss : {train_loss[-1]:0.3f} | Train_acc : {train_acc[-1]:0.2f} | Test_loss : {test_loss[-1]:0.3f} | Test_acc : {test_acc[-1]:0.2f}")