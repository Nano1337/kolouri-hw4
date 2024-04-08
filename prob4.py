import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the images
        return self.model(x)

def fgsm(input_batch, labels, model): 
    input_batch = input_batch.clone().detach()
    labels = labels.clone().detach()
    if torch.cuda.is_available():
        input_batch, labels = input_batch.to('cuda'), labels.to('cuda')

    input_batch.requires_grad = True
    output = model(input_batch)
    loss = nn.CrossEntropyLoss()

    loss_cal = loss(output, labels)
    model.zero_grad()
    loss_cal.backward()
    epsilon = 0.1
    perturbed_images = []

    for i in range(input_batch.size(0)):
        data_grad = input_batch.grad.data[i]
        sign_data_grad = data_grad.sign()
        perturbed_image = input_batch[i] - epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_images.append(perturbed_image.unsqueeze(0))

    perturbed_images = torch.cat(perturbed_images, dim=0)
    return perturbed_images

def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)
                tepoch.set_description(f"Epoch {epoch+1}")

                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

    return model

def adversarial_train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)
                tepoch.set_description(f"Epoch {epoch+1}")

                optimizer.zero_grad()
                perturbed_images = fgsm(images, labels, model)
                perturbed_output = model(perturbed_images)
                loss = criterion(perturbed_output, labels)
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

    return model

def normal_eval(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def fgsm_eval(model, test_loader): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        perturbed_image = fgsm(images, labels, model)
        perturbed_image = perturbed_image.to(device)
        output = model(perturbed_image)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Accuracy after FGSM attack: {correct / total * 100}%")

if __name__ == "__main__":
    batch_size = 1024

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    print("Begin normal training")
    model = train(model, train_loader, criterion, optimizer, epochs=10)

    accuracy = normal_eval(model, test_loader)
    print(f"Normal accuracy on the test set: {accuracy*100}%")
    
    print("Begin FGSM attack")
    fgsm_eval(model, test_loader)
    
    print("Perform adversarial training")
    model = adversarial_train(model, train_loader, criterion, optimizer, epochs=10)

    print("Normal accuracy after adversarial training")
    eval_accuracy = normal_eval(model, test_loader)
    print(f"Normal accuracy on the test set: {eval_accuracy*100}%")

    print("Begin FGSM attack")
    fgsm_eval(model, test_loader)
