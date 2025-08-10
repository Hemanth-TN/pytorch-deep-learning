
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from torchmetrics import Accuracy
import argparse


data_path = Path("./exercise_data")
image_path = data_path / "pizza_steak_sushi"

train_dir = image_path / "train"
test_dir = image_path / "test"

simple_transform = transforms.Compose([transforms.Resize(size=(64,64)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(15),
                                       transforms.ToTensor()])
#creating the datasets
train_dataset = datasets.ImageFolder(root=train_dir,
                                     transform=simple_transform,
                                     target_transform=None)

test_dataset = datasets.ImageFolder(root=test_dir,
                                    transform=simple_transform,
                                    target_transform=None)

device = 'cuda' if torch.cuda.is_available() else 'cpu'



#creating the tinyvgg
def train(num_epochs, batch_size, hidden_units, learning_rate):
    #defining the CNN model
    class TinyVGG(nn.Module):
        def __init__(self, input_neurons, hidden_units, output_neurons):
            super().__init__()
            self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=input_neurons,
                                                out_channels=hidden_units,
                                                kernel_size=3,
                                                padding=1,
                                                stride=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=hidden_units,
                                                out_channels=hidden_units,
                                                kernel_size=3,
                                                padding=1,
                                                stride=1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(hidden_units),
                                        nn.MaxPool2d(kernel_size=2))
            self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=hidden_units,
                                                out_channels=hidden_units,
                                                kernel_size=3,
                                                padding=1,
                                                stride=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=hidden_units,
                                                out_channels=hidden_units,
                                                kernel_size=3,
                                                padding=1,
                                                stride=1),
                                        nn.BatchNorm2d(hidden_units),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))
            self.classifier = nn.Sequential(nn.Flatten(),
                                            nn.Dropout(0.5),
                                            nn.Linear(in_features=hidden_units*16*16,
                                                    out_features=output_neurons))
        
        def forward(self, x):
            x = self.conv_1(x)
            x = self.conv_2(x)
            x = self.classifier(x)
            return x
    
    #Initializing the CNN
    model = TinyVGG(input_neurons=3,
                    hidden_units=hidden_units,
                    output_neurons=3)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    accuracy_fn = Accuracy(task='multiclass',
                           num_classes=3)
    accuracy_fn = accuracy_fn.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=learning_rate)

    
    
    #creating the dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    
    test_dataloader  = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size)
    train_acc_list, test_acc_list = [], []
    train_loss_list, test_loss_list = [],[]
    for epoch in range(num_epochs):
        model.train()
        train_loss, test_loss = 0, 0
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            logit_preds = model(X)

            optimizer.zero_grad()

            loss = loss_fn(logit_preds, y)
            train_loss += loss.item()

            accuracy_fn.update(logit_preds, y)

            loss.backward()

            optimizer.step()
        
        train_loss /= len(train_dataloader)
        train_loss_list.append(train_loss)
        train_acc = accuracy_fn.compute()
        accuracy_fn.reset()

        train_acc_list.append(train_acc)

        model.eval()
        with torch.inference_mode():
                for X, y in test_dataloader:
                        X, y = X.to(device), y.to(device)
                        logit_preds = model(X)

                        loss = loss_fn(logit_preds, y)
                        test_loss += loss.item()

                        accuracy_fn.update(logit_preds, y)
                
                test_loss /= len(test_dataloader)
                test_loss_list.append(test_loss)
                
                test_acc = accuracy_fn.compute()
                accuracy_fn.reset()

                test_acc_list.append(test_acc)
        
        print(f"Epoch: {epoch+1} | Train Acc: {train_acc:0.4f} | Train Loss: {train_loss:0.4f} | Test Acc: {test_acc:0.4f} | Test Loss: {test_loss:.4f}" )

    results={'train_loss_list': train_loss_list,
             'test_loss_list': test_loss_list,
             'train_acc_list': train_acc_list,
             'test_acc_list': test_acc_list}

    return results

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--num_epochs", type=int, help='Number of epochs', required=True)
     parser.add_argument("--batch_size", type=int, help="Batch size for training", required=True)
     parser.add_argument("--hidden_units", type=int, help="Number of hidden units", required=True)
     parser.add_argument("--learning_rate", type=float, help='Learning rate for training', required=True)
     args = parser.parse_args()
     res = train(num_epochs=args.num_epochs,
           batch_size=args.batch_size,
           hidden_units=args.hidden_units,
           learning_rate=args.learning_rate)
