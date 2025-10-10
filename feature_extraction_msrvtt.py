"""Extract ViT features for MSRVTT frame sequences."""

import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from vit import ViT
from prepare_msrvtt_dataset import MSRVTTDataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

PARAMS = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 16,
}


class FeatureExtractionDataset(torch.utils.data.Dataset):
    """Dataset wrapper around pre-rendered MSRVTT frames."""

    def __init__(self, ids, image_size):
        self.ids = ids
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        path = self.ids[index]
        image = Image.open(path).convert('RGB')
        tensor = self.transform(image)
        frame_id = path.stem
        return tensor, frame_id


class FeatureExtraction:
    """Iterate over frame folders and persist ViT features."""

    def __init__(self, input_folder: Path, output_folder: Path, model: torch.nn.Module, image_size: int, params: dict) -> None:
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model = model.eval()
        self.image_size = image_size
        self.params = params
        self.paths = sorted(input_folder.glob('*.jpg'))
        self.dataset = FeatureExtractionDataset(self.paths, image_size)
        self.loader = torch.utils.data.DataLoader(self.dataset, **self.params)
        if not self.output_folder.exists():
            self.output_folder.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        if not self.paths:
            print(f"Skipping {self.input_folder}, no frames detected.")
            return
        print(f"Extracting features from {self.input_folder} -> {self.output_folder}")
        for batch, frame_ids in tqdm(self.loader):
            batch = batch.to(device)
            with torch.no_grad():
                features = self.model(batch)
            for feature, frame_id in zip(features, frame_ids):
                torch.save(feature.cpu(), self.output_folder / f"{frame_id}.pt")
        print(f"Completed {self.input_folder}")


if __name__ == "__main__":
    IMAGE_SIZE = 224
    model = ViT().to(device)

    dataset = MSRVTTDataset()
    pairs = [
        (dataset.train_frame_folder, dataset.train_visual_features_folder),
        (dataset.test_frame_folder, dataset.test_visual_features_folder),
    ]

    for frame_folder, feature_folder in pairs:
        if not frame_folder.exists():
            print(f"Frame folder {frame_folder} not found; skipping.")
            continue
        extractor = FeatureExtraction(
            input_folder=frame_folder,
            output_folder=feature_folder,
            model=model,
            image_size=IMAGE_SIZE,
            params=PARAMS,
        )
        extractor.run()
