import fiftyone as fo
import fiftyone.zoo as foz

# download train dataset
dataset = foz.load_zoo_dataset(
    name="coco-2017",
    split="train",
    dataset_dir="dataset/train",
    label_types="detections",
    classes=["person"],
    max_samples=10000,
    download_if_necessary=True,
    persistent=True,
)

# download validation dataset
dataset = foz.load_zoo_dataset(
    name="coco-2017",
    split="validation",
    dataset_dir="dataset/valid",
    label_types="detections",
    classes=["person"],
    max_samples=5000,
    download_if_necessary=True,
    persistent=True,
)

# download test dataset
dataset = foz.load_zoo_dataset(
    name="coco-2017",
    split="test",
    dataset_dir="dataset/test",
    label_types="detections",
    classes=["person"],
    max_samples=5000,
    download_if_necessary=True,
    persistent=True,
)
