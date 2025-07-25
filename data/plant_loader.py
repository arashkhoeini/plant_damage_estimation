from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import torch
import random
from utils import palette
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from typing import List, Dict, Any, Union, Optional, Tuple


class PlantDataset(Dataset):
    """
    Dataset class for plant damage estimation.

    Supports both pre-training (unlabeled data) and fine-tuning (labeled data) modes.
    For labeled data, combines leaf segmentation and damage detection labels.

    Args:
        root (str): Root directory containing the dataset
        mean (list): Mean values for normalization
        std (list): Standard deviation values for normalization
        pretraining (bool): Whether to use unlabeled data for pre-training
        splits (list): List of data splits to use (e.g., ['fold1', 'fold2'])
        base_size (int): Base size for random scaling
        augment (bool): Whether to apply data augmentation
        val (bool): Whether this is validation data
        num_classes (int): Number of classes for segmentation
        crop_size (int): Size for random cropping
        scale (bool): Whether to apply random scaling
        flip (bool): Whether to apply random horizontal flipping
        rotate (bool): Whether to apply random rotation
        blur (bool): Whether to apply random blur
        grayscale (bool): Whether to apply random grayscale
        return_id (bool): Whether to return image IDs
    """

    def __init__(
        self,
        root: str,
        mean: List[float],
        std: List[float],
        pretraining: bool = False,
        splits: Optional[List[str]] = None,
        base_size: Optional[int] = None,
        augment: bool = True,
        val: bool = False,
        num_classes: Optional[int] = None,
        crop_size: int = 321,
        scale: bool = True,
        flip: bool = True,
        rotate: bool = False,
        blur: bool = False,
        grayscale: bool = False,
        return_id: bool = False,
    ):

        self.num_classes = num_classes

        self.root = root
        self.splits = splits
        # if pretraining then splits has to be None
        assert (self.splits is None) == pretraining
        self.pretraining = pretraining
        self.mean = mean
        self.std = std
        self.crop_size = crop_size
        if augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur
            self.grayscale = grayscale
        self.val = val
        self.image_names = []
        self.subdir_names = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.return_id = return_id

        cv2.setNumThreads(0)
        self.palette = palette.get_voc_palette(self.num_classes)

    def _set_files(self):
        if self.pretraining:
            self.image_dir = os.path.join(self.root, "unlabeled")
            list_file = os.path.join(self.image_dir, "list.txt")
            if os.path.exists(list_file):
                self.image_names = [
                    line.rstrip() for line in tuple(open(list_file, "r"))
                ]
            else:
                raise FileNotFoundError(f"List file not found: {list_file}")
        else:
            if self.splits is None:
                raise ValueError("splits cannot be None when not pretraining")
            self.image_dir = os.path.join(self.root, "labeled")
            lists = []
            # print('reading finetuning images')
            # print(self.splits)
            for split in self.splits:
                # print(f'fold{split}')
                lists.append(os.path.join(self.image_dir, split, "list.txt"))

            for i, list_file in enumerate(lists):
                if os.path.exists(list_file):
                    temp = [line.rstrip() for line in tuple(open(list_file, "r"))]
                    self.image_names.extend(temp)
                    self.subdir_names.extend([self.splits[i]] * len(temp))
                else:
                    raise FileNotFoundError(f"List file not found: {list_file}")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.image_names)

    def _load_data(self, index):
        """
        Load image and label data for a given index.

        Args:
            index (int): Index of the sample to load

        Returns:
            tuple: (image, label, image_id) where:
                - image: numpy array of the input image
                - label: numpy array of the segmentation label (None for pretraining)
                - image_id: string identifier for the image
        """
        image_name = self.image_names[index]

        if self.pretraining:
            image_path = os.path.join(self.image_dir, image_name)
        else:
            image_path = os.path.join(
                self.image_dir, self.subdir_names[index], image_name
            )

        image = np.asarray(Image.open(image_path), dtype=np.float32)

        if self.pretraining:
            label = None
        else:
            leaf_label_path = os.path.join(
                self.image_dir, self.subdir_names[index], "leaf_labels", image_name
            )
            leaf_label = np.asarray(Image.open(leaf_label_path), dtype=np.int32)
            damage_label_path = os.path.join(
                self.image_dir, self.subdir_names[index], "damage_labels", image_name
            )
            damage_label = np.asarray(Image.open(damage_label_path), dtype=np.int32)

            label = leaf_label
            label[damage_label == 255] = (
                120  # Comment this line to do only leaf segmentation
            )

        image_id = self.image_names[index].split("/")[-1].split(".")[0]
        return image, label, image_id

    def _test_augmentation(self, image):
        if self.crop_size:
            h, w, _ = image.shape
            # Scale the smaller side to crop size
            if h < w:
                h, w = (self.crop_size, int(self.crop_size * w / h))
            else:
                h, w = (int(self.crop_size * h / w), self.crop_size)

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

            # Center Crop
            h, w, _ = image.shape
            start_h = (h - self.crop_size) // 2
            start_w = (w - self.crop_size) // 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
        return image, None

    def _val_augmentation(self, image, label):
        if self.crop_size:
            h, w = label.shape
            # Scale the smaller side to crop size
            if h < w:
                h, w = (self.crop_size, int(self.crop_size * w / h))
            else:
                h, w = (int(self.crop_size * h / w), self.crop_size)

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)  # type: ignore
            label = np.asarray(label, dtype=np.int32)

            # Center Crop
            h, w = label.shape
            start_h = (h - self.crop_size) // 2
            start_w = (w - self.crop_size) // 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]
        return image, label

    def _augmentation(self, image, label=None):
        h, w, _ = image.shape
        # Scaling, we set the bigger to base size, and the smaller
        # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing
        if self.base_size:
            if self.scale:
                longside = random.randint(
                    int(self.base_size * 0.5), int(self.base_size * 2.0)
                )
            else:
                longside = self.base_size
            h, w = (
                (longside, int(1.0 * longside * w / h + 0.5))
                if h > w
                else (int(1.0 * longside * h / w + 0.5), longside)
            )
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            if label is not None:
                label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        h, w, _ = image.shape
        # Rotate the image with an angle between -180 and 180
        if self.rotate:
            angle = random.randint(-180, 180)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(
                image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR
            )  # , borderMode=cv2.BORDER_REFLECT)
            if label is not None:
                label = cv2.warpAffine(
                    label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST
                )  # ,  borderMode=cv2.BORDER_REFLECT)

        # Padding to return the correct crop size
        if self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT,
            }
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)  # type: ignore
                if label is not None:
                    label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)  # type: ignore

            # Cropping
            h, w, _ = image.shape
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            if label is not None:
                label = label[start_h:end_h, start_w:end_w]

        # Random H flip
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                if label is not None:
                    label = np.fliplr(label).copy()

        # Gaussian Blur (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(
                image,
                (ksize, ksize),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REFLECT_101,
            )

        # channel shuffling
        if False:
            rng = np.random.default_rng()
            rng.shuffle(image, axis=2)

        # Color dropout
        if False:
            if random.random() > 0.5:
                channel = np.random.randint(3)
                image[:, :, channel] = np.mean(image[:, :, channel])

        # Gray Scaling
        if self.grayscale:
            if random.random() > 0.5:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image, label

    def __getitem__(self, index: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, str],
        Dict[str, Any],
    ]:
        # try:
        image, label, image_id = self._load_data(index)
        # except OSError:
        #     print('OS ERROR')
        #     print(self.image_names[index])

        if self.pretraining:
            image1, _ = self._augmentation(image)  # CHANE THIS
            image2, _ = self._augmentation(image)  # CHANE THIS
            image1 = Image.fromarray(np.uint8(image1))
            image2 = Image.fromarray(np.uint8(image2))
            if self.return_id:
                return (
                    self.normalize(self.to_tensor(image1)),
                    self.normalize(self.to_tensor(image2)),
                    image_id,
                )
            else:
                return self.normalize(self.to_tensor(image1)), self.normalize(
                    self.to_tensor(image2)
                )
        else:

            if self.val:
                image1, label = self._val_augmentation(image, label)
            else:
                image1, label = self._augmentation(image, label)
            image2 = None
            image1 = Image.fromarray(np.uint8(image1))
            label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
            if self.return_id:
                return self.normalize(self.to_tensor(image1)), label, image_id
            else:
                return self.normalize(self.to_tensor(image1)), label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)  # type: ignore
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str


class PlantLoader(DataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        pretraining=False,
        splits=None,
        crop_size=None,
        base_size=None,
        scale=True,
        num_workers=1,
        val=False,
        num_classes=2,
        shuffle=False,
        flip=False,
        rotate=False,
        blur=False,
        grayscale=False,
        augment=False,
        val_split=None,
        return_id=False,
    ):

        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        kwargs = {
            "root": data_dir,
            "mean": self.MEAN,
            "std": self.STD,
            "augment": augment,
            "crop_size": crop_size,
            "base_size": base_size,
            "scale": scale,
            "flip": flip,
            "blur": blur,
            "rotate": rotate,
            "grayscale": grayscale,
            "return_id": return_id,
            "val": val,
            "splits": splits,
            "pretraining": pretraining,
            "num_classes": num_classes,
        }

        self.dataset = PlantDataset(**kwargs)
        self.nbr_examples = len(self.dataset)
        self.shuffle = shuffle
        if val_split:
            self.train_sampler, self.val_sampler = self._split_sampler(val_split)
        else:
            self.train_sampler, self.val_sampler = None, None

        self.init_kwargs = {
            "dataset": self.dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "num_workers": num_workers,
            "pin_memory": True,
        }

        super(PlantLoader, self).__init__(
            sampler=self.train_sampler, **self.init_kwargs
        )

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        self.shuffle = False

        split_indx = int(self.nbr_examples * split)
        np.random.seed(0)

        indxs = np.arange(self.nbr_examples)
        np.random.shuffle(indxs)
        train_indxs = indxs[split_indx:]
        val_indxs = indxs[:split_indx]
        self.nbr_examples = len(train_indxs)

        train_sampler = SubsetRandomSampler(train_indxs.tolist())
        val_sampler = SubsetRandomSampler(val_indxs.tolist())
        return train_sampler, val_sampler

    def get_val_loader(self):
        if self.val_sampler is None:
            return None
        # self.init_kwargs['batch_size'] = 1
        return DataLoader(sampler=self.val_sampler, **self.init_kwargs)


class InferenceLoader(DataLoader):
    """
    DataLoader for inference on unlabeled images.

    Args:
        data_dir (str): Directory containing images for inference
        batch_size (int): Batch size for inference
        num_workers (int): Number of worker processes
        crop_size (int): Size for center cropping (optional)
    """

    def __init__(self, data_dir, batch_size=1, num_workers=1, crop_size=None):
        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        # Get all image files from directory
        image_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")
        self.image_paths = []
        for file in os.listdir(data_dir):
            if file.lower().endswith(image_extensions):
                self.image_paths.append(os.path.join(data_dir, file))

        self.dataset = InferenceDataset(
            image_paths=self.image_paths,
            mean=self.MEAN,
            std=self.STD,
            crop_size=crop_size,
        )

        super(InferenceLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )


class InferenceDataset(Dataset):
    """
    Dataset for inference on individual images.
    """

    def __init__(self, image_paths, mean, std, crop_size=None, num_classes=3):
        self.image_paths = image_paths
        self.mean = mean
        self.std = std
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

        # Setup transforms
        transform_list = []
        if crop_size:
            transform_list.append(transforms.CenterCrop(crop_size))
        transform_list.extend([self.to_tensor, self.normalize])
        self.transform = transforms.Compose(transform_list)

        self.palette = palette.get_voc_palette(num_classes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image_id = os.path.basename(image_path).split(".")[0]

        if self.transform:
            image = self.transform(image)

        return image, image_id, image_path
