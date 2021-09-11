from torchvision import datasets
from torchvision import transforms
import wandb
import torch

# datasets.MNIST.mirrors = ["https://sagemaker-sample-files.s3.amazonaws.com/datasets/image/MNIST/"]
# def collate_fn(data):
#     print(data)
# loader = torch.utils.data.DataLoader(
#         datasets.MNIST(
#             'data',
#             train=False,
#             download=True,
#             # transform=transforms.Compose(
#             #     [transforms.ToTensor(),]
#             # ),
#         ),
#         collate_fn=lambda item: item,
#         batch_size=64,
#         shuffle=True,
# )

data = datasets.MNIST(
    'data',
    train=False,
    download=True
)
table = []
for i in range(1):
    table += [[wandb.Image(data[i][0]), data[i][1]]]
table = wandb.Table(data=table, columns=["image", "label"])