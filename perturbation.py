import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import the custom model
from Perturbation.model.mobilenetv2 import cifar100_mobilenetv2_x1_4
from Perturbation.model.repvgg import cifar100_repvgg_a2
from Perturbation.model.resnet import cifar100_resnet32, cifar100_resnet44
from Perturbation.model.shufflenetv2 import cifar100_shufflenetv2_x2_0
from Perturbation.model.vgg import cifar100_vgg13_bn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# vgg13_bn
# load_model = cifar100_vgg13_bn(pretrained=True)
# load_model_name = 'cifar100_vgg13_bn'

# resnet44
# load_model = cifar100_resnet44(pretrained=True)
# load_model_name = 'cifar100_resnet44'

# mobilenetv2_x1_4
# load_model = cifar100_mobilenetv2_x1_4(pretrained = True)
# load_model_name = 'cifar100_mobilenetv2_x1_4'

# shufflenetv2_x2_0
# load_model = cifar100_shufflenetv2_x2_0(pretrained=True)
# load_model_name = 'cifar100_shufflenetv2_x2_0'

# repvgg_a2
load_model = cifar100_repvgg_a2(pretrained=True)
load_model_name = 'cifar100_repvgg_a2'

load_model = load_model.to(device)


# Function to check the accuracy of a pretrained model
def check_accuracy(model, dataloader, device):
    length = len(dataloader.dataset)
    total_accuracy = 0

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            accuracy = (outputs.argmax(1) == targets).sum().item()
            total_accuracy += accuracy

    accuracy = total_accuracy / length
    print(f"Accuracy : {accuracy:.4f}")
    return accuracy

# Define the modified pretrained model which introduces a perturbation layer
class ModifiedModel(nn.Module):
    def __init__(self, model, pert_ratio):
        super(ModifiedModel, self).__init__()
        self.model = model
        self.pert_ratio = pert_ratio

    def forward(self, x):
        outputs = self.model(x)
        # Apply perturbation: randomly zero out some correct predictions
        batch_size = outputs.size(0)
        pert_mask = torch.rand(batch_size) < self.pert_ratio
        max_indices = outputs.argmax(dim=1)
        perturbed_outputs = outputs.clone()

        for i in range(batch_size):
            if pert_mask[i] == 0:
                perturbed_outputs[i, max_indices[i]] = float('-inf')

        return perturbed_outputs

if __name__ == '__main__':
    # Load the CIFAR100 test dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    data_test = torchvision.datasets.CIFAR100('dataset', train=False, download=True, transform=transform)
    dataloader = DataLoader(data_test, batch_size=64)

    # Define and load the  model for CIFAR100
    model = load_model
    print("Model successfully loaded:", model)

    # Check the accuracy of the original pretrained model
    print("Original Model:")
    check_accuracy(model, dataloader, device)

    # Record
    writer = SummaryWriter('pert_logs')
    # Modify the pretrained model based on pert_ratio
    for i in range(101):
        un_pert_ratio = round(1 - 0.01 * i, 3)
        print(f"The current pert_ratio: {un_pert_ratio}")
        modified_model = ModifiedModel(model, un_pert_ratio)

        # Check the accuracy of the modified pretrained model
        print("Modified Model:")
        pert_accuracy = check_accuracy(modified_model, dataloader, device)
        writer.add_scalar(load_model_name, pert_accuracy, i)
        print(" ")

    writer.close()