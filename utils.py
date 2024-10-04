import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def get_data_loaders(batch_size=128, num_workers=4, dataset='CIFAR10'):
    """Returns training and test loaders for the specified dataset."""
    
    if dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    else:
        raise NotImplementedError(f"Dataset {dataset} not supported yet.")

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def save_model(model, path='models/resnet34.pth'):
    """Saves the model to the specified path."""
    torch.save(model.state_dict(), path)


def load_model(model, path='models/resnet34.pth'):
    """Loads the model from the specified path."""
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    return model
