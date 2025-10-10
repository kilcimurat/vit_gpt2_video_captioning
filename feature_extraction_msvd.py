"""Extract patch features for MSVD frames using the shared ViT backbone."""

import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from vit import ViT
from prepare_msvd_dataset import MSVDDataset

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters tuned for local workstations; tweak if you see data-loader stalls.
PARAMS = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 16,
}


class FeatureExtractionDataset(torch.utils.data.Dataset):
    """Iterate over pre-extracted video frames."""

    def __init__(self, ids, im_size):
        self.ids = ids
        self.transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        path = self.ids[index]
        image = Image.open(path)
        image = image.convert('RGB')
        tensor = self.transform(image)
        frame_id = path.stem  # preserve full stem for downstream naming
        return tensor, frame_id


class FeatureExtraction:
    """Wrap feature extraction so we can reuse the ViT across folders."""

    def __init__(self, input_folder: Path, output_folder: Path, model: torch.nn.Module, image_size: int, params: dict) -> None:
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model = model.eval()
        self.image_size = image_size
        self.params = params
        self.ids = sorted(input_folder.glob('*.jpg'))
        self.dataset = FeatureExtractionDataset(self.ids, self.image_size)
        self.loader = torch.utils.data.DataLoader(self.dataset, **self.params)
        if not self.output_folder.exists():
            self.output_folder.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        if not self.ids:
            print(f"Skipping {self.input_folder}, no frames found.")
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
    model = ViT()
    model = model.to(device)

    dataset = MSVDDataset()

    pairs = [
        (dataset.train_frame_folder, dataset.train_visual_features_folder),
        (dataset.test_frame_folder, dataset.test_visual_features_folder),
    ]

    for input_folder, output_folder in pairs:
        if not input_folder.exists():
            print(f"Frame folder {input_folder} not found; skipping.")
            continue
        extractor = FeatureExtraction(
            input_folder=input_folder,
            output_folder=output_folder,
            model=model,
            image_size=IMAGE_SIZE,
            params=PARAMS,
        )
        extractor.run()
