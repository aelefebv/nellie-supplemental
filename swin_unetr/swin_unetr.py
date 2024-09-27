import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff
from monai.transforms import (
    Compose, ScaleIntensityd, RandSpatialCropd, RandFlipd,
    RandRotate90d, ToTensord, EnsureChannelFirstd, SpatialPadd
)
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from tqdm import tqdm
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from datetime import datetime
import csv

dt = datetime.now().strftime("%Y%m%d_%H%M%S")


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_tiff_image(path):
    """Load a TIFF image from the specified path."""
    image = tiff.imread(path)
    # if the image is > 3D, only take the last 3 dimensions
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    if len(image.shape) == 5:
        image = image[0, 0, :, :, :]
    return image


def normalize_image(image):
    """Normalize the image to have intensity values between 0 and 1."""
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())
    return image


class MitochondriaDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_tiff_image(self.image_paths[idx])
        mask = load_tiff_image(self.mask_paths[idx]) > 0

        image = normalize_image(image)
        mask = mask.astype(np.float32)

        data_dict = {"image": image, "label": mask}

        if self.transforms:
            data_dict = self.transforms(data_dict)

        return data_dict["image"], data_dict["label"]

# Update model initialization
model = SwinUNETR(
    img_size=(32, 128, 128),  # Update image size
    in_channels=1,
    out_channels=1,
    feature_size=48,
    # depths=(2, 2, 2),  # Reduced number of layers
    # num_heads=(3, 6, 12),
    # patch_size=(2, 4, 4),  # Adjusted patch size
    # window_size=(5, 7, 7),
    # use_checkpoint=True,
).to(device)


def train_model(image_dir, mask_dir, model_name):
    # Get lists of all image and mask file paths
    image_paths = sorted(
        [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.ome.tif')]
    )
    mask_paths = sorted(
        [os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith('instance_label.ome.tif')]
    )

    # Split the data into training and validation sets
    from sklearn.model_selection import train_test_split

    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    # Training transforms
    train_transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        ScaleIntensityd(keys="image"),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=(32, 128, 128),  # Pad Z to 32
            method='symmetric',
            mode='constant',
            value=0,
        ),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=(32, 128, 128),
            random_size=False
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[2]),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        ToTensord(keys=["image", "label"])
    ])

    # Validation transforms
    val_transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        ScaleIntensityd(keys="image"),
        ToTensord(keys=["image", "label"])
    ])

    # Create datasets
    train_dataset = MitochondriaDataset(train_image_paths, train_mask_paths, transforms=train_transforms)
    val_dataset = MitochondriaDataset(val_image_paths, val_mask_paths, transforms=val_transforms)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Loss function
    loss_function = DiceCELoss(sigmoid=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    max_epochs = 100
    val_interval = 5  # Validate every n epochs
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    csv_filename = f"training_metrics_{dt}.csv"

    # Initialize lists to store metrics
    training_losses = []
    validation_dice_scores = []
    validation_losses = []

    # Remove the CSV file if it already exists to start fresh
    if os.path.exists(csv_filename):
        os.remove(csv_filename)

    # Define the CSV header
    fieldnames = ['epoch', 'training_loss', 'validation_loss', 'validation_dice']

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"Epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        training_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Initialize validation loss
        val_epoch_loss = None

        if (epoch + 1) % val_interval == 0:

            post_trans = Compose([
                AsDiscrete(threshold=0.5)
            ])

            model.eval()
            metric = DiceMetric(include_background=True, reduction="mean")
            val_loss = 0
            val_step = 0
            with torch.no_grad():
                for val_data in val_loader:
                    val_step += 1
                    val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = sliding_window_inference(
                        inputs=val_inputs,
                        roi_size=(32, 128, 128),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.25
                    )
                    val_outputs = torch.sigmoid(val_outputs)
                    val_outputs = post_trans(val_outputs)
                    val_labels_post = post_trans(val_labels)
                    # Calculate validation loss
                    val_loss += loss_function(val_outputs, val_labels).item()
                    metric(y_pred=val_outputs, y=val_labels_post)
                val_loss /= val_step
                val_epoch_loss = val_loss
                validation_losses.append(val_epoch_loss)
                metric_score = metric.aggregate().item()
                metric.reset()
                metric_values.append(metric_score)
                validation_dice_scores.append(metric_score)
                if metric_score > best_metric:
                    best_metric = metric_score
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_name)
                    print("Saved new best metric model")
                print(
                    f"Current epoch: {epoch + 1} Current mean Dice: {metric_score:.4f} "
                    f"Best mean Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
        else:
            # Append None for epochs without validation
            validation_losses.append(None)
            validation_dice_scores.append(None)

        # Open CSV file in append mode and write metrics
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Write header only if file is new
            if epoch == 0:
                writer.writeheader()
            writer.writerow({
                'epoch': epoch + 1,
                'training_loss': epoch_loss,
                'validation_loss': val_epoch_loss,
                'validation_dice': validation_dice_scores[-1]
            })

    print(f"Training completed, best metric: {best_metric:.4f} at epoch {best_metric_epoch}")


def infer_image(image_path, model, roi_size=(32, 128, 128), sw_batch_size=4, overlap=0.25):
    model.eval()
    image = load_tiff_image(image_path)
    image = normalize_image(image)
    image = image[np.newaxis, ...]  # Add channel dimension
    # if the image is 3d, add another dimension
    if len(image.shape) == 3:
        image = image[np.newaxis, ...]
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap
        )
        output = torch.sigmoid(output)
        predicted_mask = (output > 0.5).float()

    predicted_mask = predicted_mask.cpu().numpy()[0, 0]
    return predicted_mask


def run_inference(raw_im_path, model_path, append_name):
    # Load the best model weights
    model.load_state_dict(torch.load(model_path))

    # Perform inference
    predicted_mask = infer_image(raw_im_path, model)
    raw_im_name = os.path.basename(raw_im_path)
    raw_im_dir = os.path.dirname(raw_im_path)
    pred_dir = os.path.join(raw_im_dir, 'predicted_masks')
    os.makedirs(pred_dir, exist_ok=True)
    pred_name = f"pred-{append_name}-{raw_im_name}"

    # Save the predicted mask
    tiff.imwrite(os.path.join(pred_dir, pred_name), (predicted_mask * 255).astype(np.uint8))  # Convert mask to uint8


def run_all_inferences(raw_im_list, model_list):
    for raw_im_path in raw_im_list:
        for model_path in model_list:
            model_basename = os.path.basename(model_path).split('_')[0]
            run_inference(raw_im_path, model_path, model_basename)


if __name__ == "__main__":
    train = False
    if train:
        image_dir = r'D:\test_files\aics_dataset\combo'
        mask_dir = rf'{image_dir}\nellie_output\nellie_necessities'
        organelle = os.path.basename(image_dir)
        model_name = f"{organelle}_model_{dt}.pth"
        train_model(image_dir, mask_dir, model_name)
    else:
        # raw_im_path = r"D:\test_files\aics_dataset\mito\3500000961_100X_20170609_2-Scene-09-P30-E07_mito.ome.tif"
        # model_path = r'C:\Users\austin\GitHub\nellie-supplemental\dl_examples\SWIN_3d\mito_model_20240924_211411.pth'
        # run_inference(raw_im_path, model_path, 'mito_model')
        # im_list = [
        #     r"D:\test_files\aics_dataset\combo\nellie_output\nellie_necessities\3500001085_100X_20170719_1-Scene-5-P5-E04_desmosome.ome-ZYX-Z0p29_Y0p108_X0p108-ch0.ome.tif",
        # ]
        im_dir = r'D:\test_files\nellie_revision_stuff\nellie_output\nellie_necessities\to_predict'
        im_list = [os.path.join(im_dir, f) for f in os.listdir(im_dir) if f.endswith('ome.tif')]
        print(im_list)
        model_list = [
            r'C:\Users\austin\GitHub\nellie-supplemental\dl_examples\SWIN_3d\combo_model_20240925_094755.pth',
            r'C:\Users\austin\GitHub\nellie-supplemental\dl_examples\SWIN_3d\desmosome_model_20240924_235535.pth',
            r'C:\Users\austin\GitHub\nellie-supplemental\dl_examples\SWIN_3d\mito_model_20240924_211411.pth'
        ]
        run_all_inferences(im_list, model_list)
