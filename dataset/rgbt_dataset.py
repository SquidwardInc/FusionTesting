import torch
import os
from torch.utils.data import Dataset

class RGBTinyDataset(Dataset):

    def __init__(self, rgb_dir, ir_dir, annotations, transform=None):

        self.rgb_dir = rgb_dir
        self.ir_dir = ir_dir
        self.annotations = annotations
        self.transform = transform
        self.ids = list(annotations.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        img_id = self.ids[idx]

        rgb_path = os.path.join(self.rgb_dir, img_id + ".jpg")
        ir_path = os.path.join(self.ir_dir, img_id + ".jpg")

        rgb = cv2.imread(rgb_path)
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

        boxes = self.annotations[img_id]["boxes"]
        labels = self.annotations[img_id]["labels"]

        if self.transform:
            transformed = self.transform(image=rgb, mask=ir)
            rgb = transformed["image"]
            ir = transformed["mask"]

        rgb = torch.tensor(rgb).permute(2,0,1).float() / 255
        ir = torch.tensor(ir).unsqueeze(0).float() / 255

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels)
        }

        return rgb, ir, target