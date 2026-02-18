import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Force to 1 channel (Luminance)
    transforms.ToTensor(),                       # Convert to [0, 1]
    transforms.Normalize((0.5,), (0.5,))       # UNCOMMENT THIS later for training
])

# Load CIFAR-100
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
# 3. Download and load the test data
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

print(f"Successfully loaded {len(trainset)} training images.")

classes = trainset.classes

def show_images_with_captions(images, labels):
    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        
        # Un-normalize for display
        img = images[i] * 0.5 + 0.5
        img = img.numpy().squeeze()
        
        # Create a "Prompt"
        label_name = classes[labels[i]].replace('_', ' ')
        caption = f"A photo of a {label_name}"
        
        plt.imshow(img, cmap='gray')
        plt.title(caption, fontsize=10)
        plt.axis('off')
    plt.show()

# Get a batch and display
dataiter = iter(trainloader)
images, labels = next(dataiter)
show_images_with_captions(images, labels)
