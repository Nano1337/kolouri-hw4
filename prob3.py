import torch
import torch.nn as nn
import urllib

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

def visualize_result(original_image, perturbed_image, original_class, perturbed_class, method):
    # Normalize and clip images
    def process_image(image):
        return np.clip(image * np.array([0.229, 0.224, 0.225])[:, None, None] + np.array([0.485, 0.456, 0.406])[:, None, None], 0, 1)

    original_image, perturbed_image = map(process_image, [original_image, perturbed_image])

    # Visualization
    fig, ax = plt.subplots(1, 2, tight_layout=True)
    for a, img, title in zip(ax, [original_image, perturbed_image], [f"Original Image: {original_class}", f"Perturbed Image: {perturbed_class}"]):
        a.imshow(np.transpose(img, (1, 2, 0)))
        a.set_title(title)
        a.axis('off')
    plt.savefig(f"{method}_result.png")

def fgsm(original_class, input_batch, model): 
    input_batch = input_batch.clone()
    label = torch.tensor([original_class])
    if torch.cuda.is_available():
        label = label.to('cuda')

    input_batch.requires_grad = True
    output = model(input_batch)
    loss = nn.CrossEntropyLoss()

    loss_cal = loss(output, label)
    model.zero_grad()
    loss_cal.backward()
    data_grad = input_batch.grad.data

    # Call FGSM Attack for untargeted attack by subtracting the gradient
    epsilon = 0.05
    sign_data_grad = data_grad.sign()
    perturbed_image = input_batch - epsilon * sign_data_grad  
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    output = model(perturbed_image)

    visualize_result(input_batch[0].detach().cpu().numpy(), perturbed_image[0].detach().cpu().numpy(), categories[original_class], categories[output.argmax().item()], method="FGSM")

def least_likely(original_class, input_batch, model): 
    input_batch = input_batch.clone().detach().requires_grad_(True)
    output = model(input_batch)
    # Get the least likely class instead of the original class
    least_likely_label = output.argmin(dim=1)
    if torch.cuda.is_available():
        least_likely_label = least_likely_label.to('cuda')

    input_batch.requires_grad = True
    # Use the least likely class label for loss calculation
    loss = nn.CrossEntropyLoss()
    loss_cal = loss(output, least_likely_label)
    model.zero_grad()
    loss_cal.backward()
    data_grad = input_batch.grad.data

    # Call FGSM Attack for targeted attack by adding the gradient
    epsilon = 0.05
    sign_data_grad = data_grad.sign()
    perturbed_image = input_batch + epsilon * sign_data_grad  
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    output = model(perturbed_image)

    visualize_result(input_batch[0].detach().cpu().numpy(), perturbed_image[0].detach().cpu().numpy(), categories[original_class], categories[output.argmin().item()], method="Least Likely")

def projected_gradient_descent(original_class, input_batch, model, epsilon=0.05, alpha=0.01, num_iter=10):
    perturbed_image = input_batch.clone()
    for _ in range(num_iter):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = nn.CrossEntropyLoss()
        label = torch.tensor([original_class])
        if torch.cuda.is_available():
            label = label.to('cuda')
        loss_cal = loss(output, label)
        model.zero_grad()
        loss_cal.backward()
        data_grad = perturbed_image.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_image = perturbed_image + alpha * sign_data_grad
        # Correct projection step: ensure the perturbation does not exceed epsilon
        perturbation = torch.clamp(perturbed_image - input_batch, -epsilon, epsilon)
        perturbed_image = torch.clamp(input_batch + perturbation, 0, 1).detach()
    output = model(perturbed_image)
    visualize_result(input_batch[0].detach().cpu().numpy(), perturbed_image[0].detach().cpu().numpy(), categories[original_class], categories[output.argmax().item()], method="PGD")

def carlini_wagner(original_class, input_batch, model, confidence=0, c=0.1, learning_rate=0.01, binary_search_steps=9, max_iterations=10000):
    perturbation = torch.zeros_like(input_batch, requires_grad=True)
    optimizer = torch.optim.Adam([perturbation], lr=learning_rate)
    
    original_class = torch.tensor([original_class], dtype=torch.long)
    if torch.cuda.is_available():
        original_class = original_class.cuda()

    for step in range(binary_search_steps):
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            perturbed_input = input_batch + perturbation
            perturbed_input = torch.clamp(perturbed_input, 0, 1)
            output = model(perturbed_input)
            
            # Calculate the loss using the Carlini-Wagner loss function
            real = torch.max(output * (1 - original_class))
            other = torch.max((1 - output) * original_class + output - output * original_class)
            loss = torch.clamp(real - other + confidence, min=0)
            loss = c * loss + torch.norm(perturbation)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if loss.item() < 1e-6:
                break
        
        if loss.item() > 1e-6:
            c *= 10
        else:
            c /= 10
    
    perturbed_input = input_batch + perturbation
    perturbed_input = torch.clamp(perturbed_input, 0, 1)
    
    output = model(perturbed_input)
    visualize_result(input_batch[0].detach().cpu().numpy(), perturbed_input[0].detach().cpu().numpy(), categories[original_class], categories[output.argmax().item()], method="Carlini-Wagner")

if __name__ == "__main__":
    # load pretrained model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.eval()

    # load example image 
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    top5_prob, top5_catid = torch.topk(nn.functional.softmax(output[0], dim=0), 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
    original_class = top5_catid[0].item()

    # Perform adversarial attacks
    fgsm(original_class, input_batch, model)
    least_likely(original_class, input_batch, model)
    projected_gradient_descent(original_class, input_batch, model)
    carlini_wagner(original_class, input_batch, model)