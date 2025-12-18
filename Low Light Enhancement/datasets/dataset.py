import os
import torch
import torch.utils.data
from PIL import Image
import random
from datasets.data_augment import PairCompose, PairToTensor, PairRandomHorizontalFilp


class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        train_dataset = AllWeatherDataset(self.config.data.data_dir,
                                          patch_size=self.config.data.patch_size,
                                          filelist='{}_train.txt'.format(self.config.data.train_dataset))
        val_dataset = AllWeatherDataset(self.config.data.data_dir,
                                        patch_size=self.config.data.patch_size,
                                        filelist='{}_val.txt'.format(self.config.data.val_dataset), train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, filelist=None, train=True):
        super().__init__()

        self.dir = dir
        self.file_list = filelist
        self.train = train
        self.patch_size = patch_size

        self.train_list = os.path.join(dir, self.file_list)
        with open(self.train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]

        self.input_names = input_names

        if train:
            self.transforms = PairCompose([
                PairRandomHorizontalFilp(),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])

        print(f"[Dataset] Loaded {len(self.input_names)} pairs from {self.train_list}, "
              f"patch_size={self.patch_size}, train={self.train}")

    def _crop_pair(self, low_img, high_img):
        """
        Crop a patch_size x patch_size patch from both images.
        Random crop for train, center crop for val.
        """
        ps = self.patch_size
        w, h = low_img.size

        # If image smaller than patch, resize up
        if w < ps or h < ps:
            low_img = low_img.resize((ps, ps), Image.BICUBIC)
            high_img = high_img.resize((ps, ps), Image.BICUBIC)
            return low_img, high_img

        if self.train:
            # random crop
            left = random.randint(0, w - ps)
            top = random.randint(0, h - ps)
        else:
            # center crop
            left = (w - ps) // 2
            top = (h - ps) // 2

        box = (left, top, left + ps, top + ps)
        low_img = low_img.crop(box)
        high_img = high_img.crop(box)
        return low_img, high_img

    def get_images(self, index):
        input_name = self.input_names[index].strip()

        parts = input_name.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid line in filelist: '{input_name}'")

        low_img_name, high_img_name = parts[0], parts[1]

        img_id = os.path.basename(low_img_name)

        low_img = Image.open(low_img_name).convert("RGB")
        high_img = Image.open(high_img_name).convert("RGB")

        low_img, high_img = self._crop_pair(low_img, high_img)

        low_img, high_img = self.transforms(low_img, high_img)

        return torch.cat([low_img, high_img], dim=0), img_id

    def __getitem__(self, index):
        return self.get_images(index)

    def __len__(self):
        return len(self.input_names)
