import os
from typing import Union, Tuple, Any

import numpy as np
import torch
import nibabel as nib
import torchvision
from matplotlib import pyplot as plt
from monai.transforms import RandCropByLabelClassesd
from torch import Tensor
from torch.utils.data import Dataset

import typing

# if typing.TYPE_CHECKING:
from jaxtyping import Float


class ACDC_ShortAxisDataset(Dataset):
    def __init__(self, img_dir: str, img_size:int, mean: float = -1, std: float = -1,
                 transforms: torchvision.transforms = None, max_samples: int = -1):
        """
        ACDC_ShortAxis dataset class for loading and preprocessing ACDC short-axis cardiac MRI images and labels.

        Args:
            img_dir (str): The directory path where the image files are stored.
            mean (float, optional): Mean value for image normalization. If not provided, it will be calculated from the images. Must be provided for validation and test. Defaults to -1.
            std (float, optional): Standard deviation value for image normalization. If not provided, it will be calculated from the images. Must be provided for validation and test. Defaults to -1.
            transforms (bool, optional): Flag indicating whether to apply data augmentation. Defaults to False.
            max_samples (int, optional): Maximum number of samples to include in the dataset. Defaults to -1 (include all samples).

        Raises:
            NotImplementedError: Raised if augmentation is set to True (not implemented yet).
        """
        self.transforms = transforms

        imgs = []
        labels = []
        slice_paths = []
        patient_ids = [folder for folder in os.listdir(img_dir) if os.path.isdir(f"{img_dir}/{folder}")]

        for i, patient_id in enumerate(patient_ids):
            # convert id to string with three digits
            # iterate through every image in image directory
            for img_file in os.listdir(os.path.join(img_dir, patient_id)):
                # if name ends with .nii.gz and does not contain _gt and does not contain 4d, it is an image
                if img_file.endswith(".nii.gz") and "_gt" not in img_file and "4d" not in img_file:
                    # Load image
                    img = nib.load(f"{img_dir}/{patient_id}/{img_file}").get_fdata()
                    # Load label
                    label = nib.load(f"{img_dir}/{patient_id}/{img_file.split('.')[0]}_gt.nii.gz").get_fdata()
                    # Convert to torch tensor
                    img = torch.from_numpy(img).float()
                    label = torch.from_numpy(label).float()

                    # Changing to the size specified by the input
                    crop = RandCropByLabelClassesd(
                        keys=["img", "label"],
                        label_key="label",
                        spatial_size=(img_size, img_size, img.shape[2]),
                        ratios=[0, 1, 1, 1],
                        num_classes=4,
                        num_samples=3,
                    )

                    data = crop({"img": img.unsqueeze(0), "label": label.unsqueeze(0)})

                    # Append each crop and slice to list
                    for crop in data:
                        img = crop["img"][0]
                        label = crop["label"][0]

                        # Move slices dim to the front
                        img = img.permute(2, 0, 1)
                        label = label.permute(2, 0, 1)

                        # convert to 3d tensor to list of 2d slices
                        for i in range(img.shape[0]):
                            img_slice = img[i]
                            label_slice = label[i]

                            # Convert label to int
                            label_slice = label_slice.long()

                            # Convert to one-hot encoding with class dim first
                            label_slice = torch.nn.functional.one_hot(label_slice, num_classes=4)
                            label_slice = label_slice.permute(2, 0, 1)

                            # Put into dataset
                            imgs.append(img_slice)
                            labels.append(label_slice)
                            slice_paths.append(f"{img_file.replace('.nii.gz', '')}_{i}")

        # Convert lists to torch tensors
        self.imgs = torch.stack(imgs)
        self.labels = torch.stack(labels)
        self.slice_paths = slice_paths

        if max_samples > 0:
            self.imgs = self.imgs[:max_samples]
            self.labels = self.labels[:max_samples]

        # Normalize images (if mean and std are not given, calculate them from the images. This should only be done for the training set)
        if mean == -1:
            self.mean = self.imgs.mean().item()
        else:
            self.mean = mean
        if std == -1:
            self.std = self.imgs.std().item()
        else:
            self.std = std

        self.imgs = (self.imgs - self.mean) / self.std

        # Add channel dim
        self.imgs = self.imgs.unsqueeze(1)

        # Make 3 channels
        self.imgs = self.imgs.repeat(1, 3, 1, 1)

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.imgs)

    def get_mean_std(self) -> tuple[float, float]:
        """
        Returns the mean and standard deviation of the dataset.

        Returns:
            tuple[float, float]: A tuple containing the mean and standard deviation.
        """
        return self.mean, self.std

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, str]:
        """
        Retrieve and preprocess the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[Float[torch.Tensor, "channel *spatial_dimensions"]]: A tuple containing the preprocessed sample and its corresponding label.
        """

        # We return a tuple to enable easy adaptation of the diffusion model code.
        pair = (self.imgs[idx], self.labels[idx], self.slice_paths[idx])
        if self.transforms is not None:
            img = self.transforms(self.imgs[idx])
            label = self.transforms(self.labels[idx])
            return img, label, self.slice_paths[idx]
        return pair


def visualize_image_seg_pair(data_tuple):
    # img = data_tuple[0].squeeze(0).permute(1, 2, 0).numpy()
    img = data_tuple[0].squeeze(0).mean(dim=0).numpy()
    # We convert the mask into actual integer values
    mask = torch.argmax(data_tuple[1].squeeze(0), dim=0)
    mask = mask.numpy()
    print(f"Image slice is {data_tuple[2]}")
    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original image
    im = ax[0].imshow(img, cmap='gray', vmin=img.min(), vmax=img.max())
    ax[0].set_title('Original Image')
    ax[0].axis('off')  # Hide the axis
    fig.colorbar(im, ax=ax[0])  # Add colorbar to indicate the value range

    # Create an RGB image for the segmentation
    height, width = img.shape[-2], img.shape[-1]
    segmentation_rgb = np.zeros((height, width, 3))

    # Overlay the segmentation labels
    for c in range(3):  # RGB channels
        segmentation_rgb[:, :, c] = ((mask - 1) == c) * 255
    segmentation_rgb = segmentation_rgb.astype(int)
    # Plot the segmentation mask
    ax[1].imshow(segmentation_rgb)
    ax[1].set_title('Multi-Label Segmentation')
    ax[1].axis('off')  # Hide the axis

    # Display the plots
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    img_dir = '/mnt/elephant/chinmay/ACDC/database/training/'
    patient_ids = [1, 2]
    dataset = ACDC_ShortAxisDataset(img_dir=img_dir, img_size=128)
    # print(dataset[0]["img"].shape)
    print(dataset[0][1].min(), dataset[0][0].max())
    # print(dataset.get_mean_std())
    visualize_image_seg_pair(data_tuple=dataset[5])
