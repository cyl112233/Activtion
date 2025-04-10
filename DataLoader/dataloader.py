import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(Data, batch_size,file):

    trans = transforms.Compose([transforms.ToTensor(),transforms.Resize(32)])
    if Data == 'MNIST':
        train_loader = torchvision.datasets.MNIST(root=file, train=True, transform=trans, download=True)
        test_loader = torchvision.datasets.MNIST(root=file, train=False, transform=trans, download=True)
        num = 10
    if Data == 'CIFAR10':
        train_loader = torchvision.datasets.CIFAR10(root=file, train=True, transform=trans,download=True)
        test_loader = torchvision.datasets.CIFAR10(root=file, train=False,transform=trans, download=True)
        num = 10
    if Data == 'CIFAR100':
        train_loader = torchvision.datasets.CIFAR100(root=file, train=True, transform=trans, download=True)
        test_loader = torchvision.datasets.CIFAR100(root=file, train=False, transform=trans, download=True)
        num = 100
    if Data == 'FashionMNIST':
        train_loader = torchvision.datasets.FashionMNIST(root=file, train=True, transform=trans,download=True)
        test_loader = torchvision.datasets.FashionMNIST(root=file, train=False,transform=trans, download=True)
        num = 10
    if Data == 'ImageNet':
        train_loader = torchvision.datasets.ImageNet(root=file, train=True, transform=trans,download=True)
        test_loader = torchvision.datasets.ImageNet(root=file, train=False,transform=trans, download=True)
        num = 1000

    train = DataLoader(train_loader, batch_size=batch_size, shuffle=True, num_workers=0)
    test = DataLoader(test_loader, batch_size=batch_size, shuffle=True, num_workers=0)
    return train, test, [train.__len__(), test.__len__()],num


