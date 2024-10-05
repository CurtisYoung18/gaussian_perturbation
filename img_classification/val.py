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

# ResNet56 for CIFAR-100
load_model = cifar100_resnet56(pretrained=True)
load_model_name = 'resnet56_cifar100'

# ResNet56 for CIFAR-10
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
        pert_ratio (float): The perturbation ratio determining the noise's standard deviation.
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

        # Directly set noise standard deviation to perturbation ratio
        noise_std = self.pert_ratio

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

    # Open a file to save the results with header
    with open('results.txt', 'w') as results_file:
        # 写入表头
        results_file.write("std\taccuracy\n")

        # Define a list of perturbation ratios (standard deviations)
        pert_ratio_list = [round(0.2 * i, 1) for i in range(101)]  # 0.0, 0.2, ..., 20.0

        for idx, pert_ratio in enumerate(pert_ratio_list):
            print(f"Perturbation ratio (std): {pert_ratio}")

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

            # Write the results to the file
            results_file.write(f"{pert_ratio}\t{pert_accuracy:.4f}\n")

    writer.close()
    print("Results have been saved to 'results.txt'.")