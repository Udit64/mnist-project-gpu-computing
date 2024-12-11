import os
from torchvision import datasets, transforms

# Set the directory to save the dataset
current_directory = os.path.dirname(os.path.abspath(__file__))  # Current file directory
data_path = os.path.join(current_directory, "data/images")

# Ensure the directory exists
os.makedirs(data_path, exist_ok=True)

# Define a transform to convert the data to tensors (optional)
transform = transforms.Compose([transforms.ToTensor()])

try:
    # Download the MNIST dataset
    mnist_data = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    print(f"MNIST dataset downloaded and saved to {data_path}")
except Exception as e:
    print("An error occurred while downloading MNIST:", str(e))
