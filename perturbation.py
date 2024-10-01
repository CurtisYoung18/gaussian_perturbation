import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50

# Import custom models
from model.resnet import cifar100_resnet56, cifar10_resnet56


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Model Reference: https://github.com/chenyaofo/pytorch-cifar-models
"""

# Select and load the pre-trained model

# ResNet56 for cifar100
load_model = cifar100_resnet56(pretrained=True)
load_model_name = 'resnet56_cifar100'

# ResNet56 for cifar10
# load_model = cifar10_resnet56(pretrained=True)
# load_model_name = 'resnet56_cifar10'


load_model = load_model.to(device)

# Freeze all model parameters to prevent training
for param in load_model.parameters():
    param.requires_grad = False

def check_accuracy(model, dataloader, device):
    """
    Evaluate the accuracy of the model on a given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): The device to perform computation on.

    Returns:
        float: The accuracy of the model on the dataset.
    """
    total_samples = len(dataloader.dataset)
    total_correct = 0

    model.eval()
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == targets).sum().item()

    accuracy = total_correct / total_samples
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


class PerturbModel(nn.Module):
    """
    A modified model that adds Gaussian noise to the logits to perturb model performance.

    Args:
        model (nn.Module): The original pre-trained model.
        pert_ratio (float): The perturbation ratio determining the noise's standard deviation as a fraction of logit magnitude.
    """

    def __init__(self, model, pert_ratio):
        super(PerturbModel, self).__init__()
        self.model = model
        self.pert_ratio = pert_ratio

    def forward(self, x):
        """
        Forward pass with added Gaussian noise to the logits.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Perturbed logits.
        """
        logits = self.model(x)

        # Scale noise relative to logit magnitudes to ensure consistent perturbation
        logit_mean = logits.abs().mean()

        # 1  use *20 on cifar10 and *10  on cifar100
        # noise_std = self.pert_ratio * 10

        # 2 use *10 on cifar10 and *5 on cifar100
        noise_std = logit_mean * self.pert_ratio * 5

        noise = torch.normal(mean=0.0, std=noise_std, size=logits.size()).to(x.device)

        # Add noise to logits
        logits_noisy = logits + noise

        return logits_noisy

if __name__ == '__main__':
    # Load CIFAR-100 test dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        ),
    ])
    test_dataset = torchvision.datasets.CIFAR100(
        root='dataset', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load CIFAR-10 test dataset
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(
    #         mean=[0.4914, 0.4822, 0.4465],
    #         std=[0.2023, 0.1994, 0.2010],
    #     ),
    # ])
    # test_dataset = torchvision.datasets.CIFAR10(
    #     root='dataset', train=False, download=True, transform=transform
    # )
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define and load the model
    model = load_model
    print("Model successfully loaded.")

    # Evaluate the original pre-trained model
    print("Original Model:")
    original_accuracy = check_accuracy(model, test_loader, device)

    # Initialize TensorBoard writer
    writer = SummaryWriter('gaussian_noise_outputs_logs')

    # Define a list of perturbation ratios (standard deviations as fractions of logit magnitudes)
    # Smaller step sizes for smoother degradation
    pert_ratio_list = [round(0.05 * i, 2) for i in range(21)]  # 0.05, 0.10, ..., 1.00

    for idx, pert_ratio in enumerate(pert_ratio_list):
        print(f"Perturbation ratio: {pert_ratio}")

        perturb_model = PerturbModel(
            model=model,
            pert_ratio=pert_ratio
        ).to(device)

        # Evaluate the modified model; gradients are not required
        print("Perturbed Model:")
        pert_accuracy = check_accuracy(perturb_model, test_loader, device)
        writer.add_scalar('resnet56_cifar100_2', pert_accuracy, idx)
        relative_ratio = pert_accuracy / original_accuracy if original_accuracy > 0 else 0.0
        print(f"Relative Accuracy Ratio: {relative_ratio:.4f}\n")

    writer.close()
