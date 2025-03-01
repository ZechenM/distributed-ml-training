from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import pickle

def prepare_mnist_data():
    # Load the MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    mnist_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Split the dataset into 3 parts
    split_sizes = [20000, 20000, 20000]  # Each worker gets 20,000 samples
    split_datasets = random_split(mnist_dataset, split_sizes)

    # Create dataloaders for each worker
    dataloaders = [DataLoader(ds, batch_size=64, shuffle=True) for ds in split_datasets]

    return dataloaders


if __name__ == "__main__":
    dataloaders = prepare_mnist_data()
    for i, dataloaders in enumerate(dataloaders):
        with open(f'dataloader_{i}.pkl', 'wb') as f:
            pickle.dump(dataloaders, f)
    print("Data prepared and split into 3 dataloaders.")
