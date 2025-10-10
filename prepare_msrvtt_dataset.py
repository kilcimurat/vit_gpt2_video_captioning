from pathlib import Path
import pathlib
from tqdm import tqdm
from typing import List, Tuple, Optional
import json
import os
import shutil
from torchvision.datasets.utils import download_url, extract_archive
from torchvision.io.video import read_video
import av


import pytube
from pytube import YouTube


from urllib.request import HTTPError, Request, urlopen
from urllib.error import URLError
from socket import gaierror
from http.client import IncompleteRead

'''
except URLError:
    print("URLError Error: " + video_id)
    stat = "URLError Error"
    return stat, False
except gaierror:
    print("gaierror Error: " + video_id)
    stat = "gaierror Error"
    return stat, False
'''
from video_utils import clip_video

from text_processing import get_captions_length
from torchtext.data.utils import get_tokenizer

import csv

# Default number for seed is 0.
# random.seed(0)

# VAL_IDS = 6513  # index number where validation data starts.


def load_json_list(json_path: pathlib.Path, video_folder: pathlib.Path, split="train") -> Tuple[List[str], List[str], List[str]]:
    # Initialize video_paths list.
    video_paths = []
    # Initialize ids list.
    video_ids = []
    # Initialize train captions list.
    captions = []
    # Load json file in train_data
    data = json.loads(json_path.read_bytes())

    for annotation in data['info']:
        if annotation['split'] == split:
            video_name = annotation['video_id']
            video_path = video_folder / f"{video_name}.mp4"
            video_paths.append(video_path)
            video_ids.append(annotation['id'])

    # Go through train data.
    print(f'Loading {str(json_path)} data...')
    for annotation in tqdm(data['sentences']):
        caption = 'boc ' + annotation['caption'] + ' eoc'
        captions.append(caption)

    print('Data is loaded.')
    
    return video_paths, captions, video_ids

class MSRVTTDataset():
    '''
        Utilities for video captioning dataset of MSRVTT

        Initialize:
        # dt = MSRVTTDataset()

        Download MSRVTT Dataset:
        # dt.download_dataset()

        Load captions, paths, times and ids:
        # train_data, test_data = dt.load_data()

        # paths, captions, ids, start_times, end_times = zip(*train_data)
        In test data captions are not important therefore only one corresponding caption for a video added.
        # paths, captions, ids, start_times, end_times = zip(*test_data)

    '''
    def __init__(self, root_folder: Optional[pathlib.Path] = None) -> None:

        # name of the dataset.
        self.name = "MSRVTT"

        env = os.getenv
        self.train_video_sources = [
            (env('MSRVTT_TRAIN_ARCHIVE_URL'), 'train_val_videos.zip'),
            ('https://huggingface.co/datasets/RangiLyu/MSRVTT/resolve/main/train_val_videos.zip?download=1', 'train_val_videos.zip'),
            ('https://huggingface.co/datasets/dqa/MSRVTT/resolve/main/train_val_videos.zip?download=1', 'train_val_videos.zip'),
            ('https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/msrvtt_train_val_videos.zip', 'msrvtt_train_val_videos.zip'),
        ]

        self.test_video_sources = [
            (env('MSRVTT_TEST_ARCHIVE_URL'), 'test_videos.zip'),
            ('https://huggingface.co/datasets/RangiLyu/MSRVTT/resolve/main/test_videos.zip?download=1', 'test_videos.zip'),
            ('https://huggingface.co/datasets/dqa/MSRVTT/resolve/main/test_videos.zip?download=1', 'test_videos.zip'),
            ('https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/msrvtt_test_videos.zip', 'msrvtt_test_videos.zip'),
        ]

        self.video_archive_repo = env('MSRVTT_ARCHIVE_REPO', 'RangiLyu/MSRVTT')

        self.annotation_sources = {
            'train': [
                env('MSRVTT_TRAIN_ANN_URL'),
                'https://dl.fbaipublicfiles.com/pvse/msrvtt/train_val_videodatainfo.json',
                'https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/train_val_videodatainfo.json',
                'https://raw.githubusercontent.com/yalesong/pvse/master/data/MSRVTT/train_val_videodatainfo.json',
                'https://huggingface.co/datasets/RangiLyu/MSRVTT/resolve/main/train_val_videodatainfo.json?download=1',
                'https://huggingface.co/datasets/dqa/MSRVTT/resolve/main/train_val_videodatainfo.json?download=1',
            ],
            'test': [
                env('MSRVTT_TEST_ANN_URL'),
                'https://dl.fbaipublicfiles.com/pvse/msrvtt/test_videodatainfo.json',
                'https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/test_videodatainfo.json',
                'https://raw.githubusercontent.com/yalesong/pvse/master/data/MSRVTT/test_videodatainfo.json',
                'https://huggingface.co/datasets/RangiLyu/MSRVTT/resolve/main/test_videodatainfo.json?download=1',
                'https://huggingface.co/datasets/dqa/MSRVTT/resolve/main/test_videodatainfo.json?download=1',
            ],
        }

        ann_repo_overrides = env('MSRVTT_ANN_HF_REPOS', '')
        self.annotation_hf_repos = [repo.strip() for repo in ann_repo_overrides.split(',') if repo and repo.strip()]
        if self.video_archive_repo and self.video_archive_repo not in self.annotation_hf_repos:
            self.annotation_hf_repos.append(self.video_archive_repo)
        for default_repo in ['dqa/MSRVTT']:
            if default_repo not in self.annotation_hf_repos:
                self.annotation_hf_repos.append(default_repo)

        # Project root Path
        default_root = Path(os.getenv('MSRVTT_ROOT', '/Volumes/KINGSTON/dataset'))
        self.root_folder = Path(root_folder) if root_folder else default_root

        # Drive Path
        self.dataset_folder = self.root_folder / self.name

        # train videos Path
        self.train_folder = self.dataset_folder / "TrainValVideo"

        # val videos Path
        self.test_folder = self.dataset_folder / "TestVideo"

        # train frames Path
        self.train_frame_folder = self.dataset_folder / 'TrainValVideoFrames'

        # train frames Path
        self.test_frame_folder = self.dataset_folder / 'TestVideoFrames'

        # initial train captions path
        self.default_train_annotations = self.dataset_folder / "train_val_videodatainfo.json"

        # initial test captions path
        self.default_test_annotations = self.dataset_folder / "test_videodatainfo.json"

        # updated train captions path
        self.train_annotations = self.dataset_folder / "train_annotations.json"

        # updated train captions path
        self.val_annotations = self.dataset_folder / "val_annotations.json"

        # updated train captions path
        self.test_annotations = self.dataset_folder / "test_annotations.json"

        # train visual features Path
        self.train_visual_features_folder = self.dataset_folder / "features_visual_train"

        # train audial features Path
        self.train_audial_features_folder = self.dataset_folder / "features_audial_train"

        # test features Path
        self.test_visual_features_folder = self.dataset_folder / "features_visual_test"

        # test features Path
        self.test_audial_features_folder = self.dataset_folder / "features_audial_test"

        # info folder Path
        self.info_folder = self.dataset_folder / "info"

        self.models_folder = self.dataset_folder / "models"

        self.tensorboard_folder = self.dataset_folder / "tensorboard_files"

        # Min and Max caption lengths which cover most of the captions. 80%
        self.min_caption_length = 4
        self.max_caption_length = 12

        # Max frames sizes for videos factored.
        self.max_frames = {
            "1": {
                "audio": 258,
                "visual": 901
            },
            "0.9": {
                "audio": 232,
                "visual": 811
            },
            "0.8": {
                "audio": 207,
                "visual": 721
            },
            "0.7": {
                "audio": 181,
                "visual": 631
            },
            "0.6": {
                "audio": 155,
                "visual": 541
            },
            "0.5": {
                "audio": 129,
                "visual": 450
            },
            "0.4": {
                "audio": 103,
                "visual": 360
            }
        }

    def download_annotations(self) -> None:
        """Download MSRVTT annotation JSON files via direct links."""

        targets = [
            ('train', self.default_train_annotations, 'train_val_videodatainfo.json'),
            ('test', self.default_test_annotations, 'test_videodatainfo.json'),
        ]

        for split, destination, filename in targets:
            if destination.exists():
                print(f"{destination.name} already exists.")
                continue
            if not destination.parent.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)

            local_env_key = f"MSRVTT_{split.upper()}_ANN_PATH"
            if self._copy_local_annotation(os.getenv(local_env_key), filename, destination):
                continue

            if self._copy_local_annotation(os.getenv('MSRVTT_ANN_DIR'), filename, destination):
                continue

            candidates = [url for url in self.annotation_sources.get(split, []) if url]
            success = False
            for url in candidates:
                try:
                    print(f"Downloading {split} annotations from {url}...")
                    self._download_to_path(url, destination)
                except Exception as exc:
                    print(f"Failed to download {split} annotations from {url}: {exc}")
                    continue
                else:
                    success = True
                    break
            if not success:
                success = self._download_annotations_from_hf(split, filename, destination)
            if not success:
                raise RuntimeError(
                    f"Unable to download {split} annotations. "
                    f"Set MSRVTT_{split.upper()}_ANN_URL to a reachable link or configure Hugging Face Hub credentials and retry."
                )

    def _download_annotations_from_hf(self, split: str, filename: str, destination: Path) -> bool:
        """Fallback: fetch annotation JSON via Hugging Face Hub if direct links fail."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("huggingface_hub not installed; skipping HF Hub fallback.")
            return False

        token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
        repos = [repo for repo in self.annotation_hf_repos if repo]
        last_error = None
        local_path = None
        for repo in repos:
            try:
                local_path = hf_hub_download(
                    repo_id=repo,
                    repo_type='dataset',
                    filename=filename,
                    token=token,
                )
                break
            except Exception as exc:
                print(f"Hugging Face Hub download failed for {split} annotations from {repo}: {exc}")
                last_error = exc
        if not local_path:
            return False

        try:
            shutil.copy(local_path, destination)
        except Exception as exc:
            print(f"Failed to copy HF Hub {split} annotations to destination: {exc}")
            return False

        print(f"Downloaded {split} annotations via Hugging Face Hub.")
        return True

    def _copy_local_annotation(self, source_hint: Optional[str], filename: str, destination: Path) -> bool:
        if not source_hint:
            return False
        source_path = Path(source_hint).expanduser()
        if source_path.is_dir():
            candidate = source_path / filename
        else:
            candidate = source_path
        if not candidate.exists():
            print(f"Local annotation source not found: {candidate}")
            return False
        try:
            shutil.copy(candidate, destination)
        except Exception as exc:
            print(f"Failed to copy local annotation {candidate}: {exc}")
            return False
        print(f"Copied {filename} from local path {candidate}.")
        return True

    def _download_to_path(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            download_url(url=url, root=str(destination.parent), filename=destination.name)
            return
        except Exception as exc:
            if destination.exists():
                destination.unlink()
            self._download_with_headers(url, destination, original_exc=exc)

    def _download_with_headers(self, url: str, destination: Path, original_exc: Exception) -> None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
        }
        request = Request(url, headers=headers)
        try:
            with urlopen(request) as response, open(destination, 'wb') as fh:
                shutil.copyfileobj(response, fh)
        except Exception as exc:
            if destination.exists():
                destination.unlink()
            raise exc from original_exc

    def download_video_archives(self) -> None:
        """Download MSRVTT video archives from direct HTTP sources when available."""

        specs = [
            ('train', self.train_folder, self.train_video_sources),
            ('test', self.test_folder, self.test_video_sources),
        ]

        for split, target_folder, candidates in specs:
            if target_folder.exists() and any(target_folder.glob('*.mp4')):
                continue
            target_folder.mkdir(parents=True, exist_ok=True)
            success = False
            expected_filename = None
            for url, filename in candidates:
                if not url:
                    continue
                expected_filename = expected_filename or filename
                archive_path = self.dataset_folder / filename
                extract_root = self.dataset_folder / f'_extracted_{split}'
                if extract_root.exists():
                    shutil.rmtree(extract_root, ignore_errors=True)
                try:
                    print(f"Downloading {split} archive from {url}...")
                    download_url(url=url, root=str(self.dataset_folder), filename=filename)
                    extract_archive(str(archive_path), str(extract_root))
                    moved = self._move_extracted_videos(extract_root, target_folder)
                    success = moved > 0
                    if not success:
                        print(f"No video files found in archive {filename}.")
                except Exception as exc:
                    print(f"Failed to download {split} archive from {url}: {exc}")
                    success = False
                finally:
                    if archive_path.exists():
                        try:
                            archive_path.unlink()
                        except FileNotFoundError:
                            pass
                    if extract_root.exists():
                        shutil.rmtree(extract_root, ignore_errors=True)
                if success:
                    break
            if not success:
                expected_filename = expected_filename or f"{split}_videos.zip"
                success = self._download_archive_from_hf(split, expected_filename, target_folder)
            if not success:
                print(f"Unable to download {split} archive via configured mirrors or Hugging Face Hub."
                      f" Set MSRVTT_{split.upper()}_ARCHIVE_URL or MSRVTT_ARCHIVE_REPO to override.")

    def _move_extracted_videos(self, source_root: Path, target_folder: Path) -> int:
        count = 0
        if not source_root.exists():
            return count
        for video_file in source_root.rglob('*.mp4'):
            destination = target_folder / video_file.name
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(video_file), str(destination))
            count += 1
        return count

    def _download_archive_from_hf(self, split: str, filename: str, target_folder: Path) -> bool:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("huggingface_hub not installed; skipping HF Hub fallback for archives.")
            return False

        token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
        try:
            local_path = hf_hub_download(
                repo_id=self.video_archive_repo,
                repo_type='dataset',
                filename=filename,
                token=token,
            )
        except Exception as exc:
            print(f"Hugging Face Hub download failed for {split} archive: {exc}")
            return False

        temp_name = f"hf_{split}_{filename}"
        temp_archive = self.dataset_folder / temp_name
        try:
            shutil.copy(local_path, temp_archive)
        except Exception as exc:
            print(f"Failed to copy HF Hub {split} archive to workspace: {exc}")
            if temp_archive.exists():
                try:
                    temp_archive.unlink()
                except FileNotFoundError:
                    pass
            return False

        extract_root = self.dataset_folder / f'_extracted_{split}'
        if extract_root.exists():
            shutil.rmtree(extract_root, ignore_errors=True)
        try:
            extract_archive(str(temp_archive), str(extract_root))
            moved = self._move_extracted_videos(extract_root, target_folder)
            if moved == 0:
                print(f"No video files found in HF Hub archive {filename}.")
                return False
        except Exception as exc:
            print(f"Failed to extract HF Hub {split} archive {filename}: {exc}")
            return False
        finally:
            if temp_archive.exists():
                try:
                    temp_archive.unlink()
                except FileNotFoundError:
                    pass
            if extract_root.exists():
                shutil.rmtree(extract_root, ignore_errors=True)

        print(f"Downloaded {split} archive via Hugging Face Hub.")
        return True

    def download_videos(self) -> None:
        '''
            Download videos from youtube into the MSRVTT folder.
        '''

        download_list = [self.default_train_annotations, self.default_test_annotations]
        download_folder_list = [self.train_folder, self.test_folder]

        for path, folder in zip(download_list, download_folder_list):
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)

            status = []
            
            with open(path, 'r') as json_path:
                data = json.load(json_path)
                for annotation in data['videos']:
                    video_id = annotation['video_id']
                    video_name = video_id + '.mp4'
                    start_time = annotation['start time']
                    end_time = annotation['end time']
                    url = annotation['url']
                    stat, is_available = self.download_video(
                        video_id=video_id,
                        start_time=start_time,
                        end_time=end_time,
                        url=url,
                        download_folder=folder,
                    )


                    status.append({'video_id': video_id, 'start_time': start_time, 'end_time': end_time, 'status': stat, 'is_available': is_available, 'url': url})
                
            # write status to json file.
            with open(str(folder.parent / f'{folder.name}_status.json'), 'w') as f:
                json.dump(status, f)

    # download youtube videos start time to end time from id.
    def download_video(self, video_id: str, start_time: float, end_time: float, url: str, download_folder: Path) -> Tuple[str, bool]:

        '''
            Download youtube videos start time to end time from id.
        '''

        yt = YouTube(url)
        try:
            download_folder.mkdir(parents=True, exist_ok=True)
            temp_basename = f"{video_id}_full"
            temp_video_path = download_folder / f"{temp_basename}.mp4"
            clipped_video_path = download_folder / f"{video_id}.mp4"
            if clipped_video_path.exists():
                print(f"Clip already exists: {clipped_video_path.name}")
                stat = "Available"
                return stat, True

            stream = yt.streams.filter(file_extension="mp4", resolution="360p").first()
            if stream is None:
                stream = yt.streams.filter(file_extension="mp4").order_by('resolution').desc().first()
            if stream is None:
                print(f"No MP4 stream available: {video_id}")
                stat = "No Stream"
                return stat, False

            stream.download(output_path=str(download_folder), filename=temp_basename)
            clip_video(video_path=temp_video_path, start_time=start_time, end_time=end_time, output_path=str(clipped_video_path))
            if temp_video_path.exists():
                temp_video_path.unlink()
            print("Downloaded: " + video_id)
            stat = "Available"
            return stat, True
        except pytube.exceptions.VideoUnavailable:
            print("Video Unavailable: " + video_id)
            stat = "Unavailable"
            return stat, False
        except KeyError:
            print("Key Error: " + video_id)
            stat = "Key Error"
            return stat, False
        except HTTPError:
            print("HTTP Error: " + video_id)
            stat = "HTTP Error"
            return stat, False
        except IncompleteRead:
            print("Incomplete Read: " + video_id)
            stat = "Incomplete Read Error"
            if temp_video_path.exists():
                temp_video_path.unlink()
            return stat, False
        except av.error.ValueError:
            print("Value Error: " + video_id)
            stat = "Value Error"
            return stat, False
      
        
    
    def download_dataset(self) -> None:
        self.download_annotations()
        self.download_video_archives()
        train_missing = not (self.train_folder.exists() and any(self.train_folder.glob('*.mp4')))
        test_missing = not (self.test_folder.exists() and any(self.test_folder.glob('*.mp4')))
        if train_missing or test_missing:
            print('Falling back to YouTube downloads for missing MSRVTT videos. This may take a long time.')
            self.download_videos()
        self.update_annotations()

    def update_annotations(self) -> None:

        # Train Val annotations
        # get video ids from trainval annotations
        print("Updating Train Val annotations...")
        train_video_ids = []
        train_ids = []
        val_video_ids = []
        val_ids = []
        train_annotations = []
        val_annotations = []
        with open(self.default_train_annotations, 'r') as json_path:
            # load json file
            data = json.load(json_path)

            for video in data['videos']:
                if video['split'] == 'train':
                    train_video_ids.append(video['video_id'])
                    train_ids.append(video['id'])
                elif video['split'] == 'validate':
                    val_video_ids.append(video['video_id'])
                    val_ids.append(video['id'])

            # update annotations if the corresponding video exists.
            train_sen_ids = []
            val_sen_ids = []
            for sentence_data in data['sentences']:
                if sentence_data['video_id'] in train_video_ids:
                    sentence_id = sentence_data['sen_id']
                    video_name = sentence_data['video_id']
                    video_path = self.train_folder / f"{video_name}.mp4"
                    if not video_path.exists():
                        continue
                    caption = sentence_data['caption']
                    video_id = int(video_name[5:])
                    train_data = {'video_id': video_id, 'video_name': video_name, 'caption': caption, 'sentence_id': sentence_id}
                    train_annotations.append(train_data)
                elif sentence_data['video_id'] in val_video_ids:
                    sentence_id = sentence_data['sen_id']
                    video_name = sentence_data['video_id']
                    video_path = self.train_folder / f"{video_name}.mp4"
                    if not video_path.exists():
                        continue
                    caption = sentence_data['caption']
                    video_id = int(video_name[5:])
                    val_data = {'video_id': video_id, 'video_name': video_name, 'caption': caption, 'sentence_id': sentence_id}
                    val_annotations.append(val_data)
        
        # write train annotations to json file.
        with open(self.train_annotations, 'w') as f:
            json.dump(train_annotations, f)

        # write val annotations to json file.
        with open(self.val_annotations, 'w') as f:
            json.dump(val_annotations, f)

        # Test annotations
        print("Updating Test annotations...")

        with open(self.default_test_annotations, 'r') as json_path:
            # load json file
            data = json.load(json_path)
            test_annotations = []
            # Check if the sentence id is repeated. Because there are some duplicate sentences in the test set. I don't know why.
            sen_ids = []
            # update annotations if the corresponding video exists.

            for sentence_data in data['sentences']:
                sentence_id = sentence_data['sen_id']
                video_name = sentence_data['video_id']
                video_path = self.test_folder / f"{video_name}.mp4"
                if not video_path.exists():
                    continue
                caption = sentence_data['caption']
                video_id = int(video_name[5:])
                test_data = {'video_id': video_id, 'video_name': video_name, 'caption': caption, 'sentence_id': sentence_id}
                test_annotations.append(test_data)

        # write test annotations to json file.
        with open(self.test_annotations, 'w') as f:
            json.dump(test_annotations, f)

    def load_annotations(self, annotations) -> Tuple[List[str], List[str], List[str]]:

        data = json.loads(annotations.read_bytes())
        names = []
        captions = []
        ids = []
        for sample in data:
            names.append(sample["video_name"])
            captions.append(sample["caption"])
            ids.append(sample["video_id"])

        return names, captions, ids

    def load_data(self) -> Tuple[List[str], List[str], List[str]]:
        '''
            Load the MSRVTT captions and their corresponding video ids.
            paths, captions, ids, start_times, end_times
        '''
        train_names, train_captions, train_ids = self.load_annotations(self.train_annotations)
        train_data = zip(train_names, train_captions, train_ids)
        val_names, val_captions, val_ids = self.load_annotations(self.val_annotations)
        val_data = zip(val_names, val_captions, val_ids)

        train_val_names = train_names + val_names
        train_val_captions = train_captions + val_captions
        train_val_ids = train_ids + val_ids

        
        train_val_data = zip(train_val_names, train_val_captions, train_val_ids)
        test_names, test_captions, test_ids = self.load_annotations(self.test_annotations)
        test_data = zip(test_names, test_captions, test_ids)

        return train_val_data, None, test_data
        #return train_data, val_data, test_data

    def get_info_of_dataset(self) -> None:
        
        if not self.info_folder.exists():
            self.info_folder.mkdir()

        caption_lengths = self.caption_lengths()
        train_video_lengths, val_video_lengths, test_video_lengths = self.video_lengths()

        # write caption_lengths to a csv file.
        with open((self.info_folder / "caption_lengths.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Length', 'Caption Lengths'])
            for length, caption_length in enumerate(caption_lengths):
                writer.writerow([length, caption_length])

        # write train video lengths to a csv file.
        with open((self.info_folder / "train_video_lengths.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Train Video Lengths'])
            for train_video_length in train_video_lengths:
                writer.writerow([train_video_length['video_id'], train_video_length['video_length']])

        # write val video lengths to a csv file.
        with open((self.info_folder / "val_video_lengths.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Val Video Lengths'])
            for val_video_length in val_video_lengths:
                writer.writerow([val_video_length['video_id'], val_video_length['video_length']])

        # write test video lengths to a csv file.
        with open((self.info_folder / "test_video_lengths.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Test Video Lengths'])
            for test_video_length in test_video_lengths:
                writer.writerow([test_video_length['video_id'], test_video_length['video_length']])

    def caption_lengths(self) -> None:
        tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        print('Analyzing caption lengths of the dataset if there is a corresponding video...')
        captions = []
        with open(self.train_annotations, 'r') as json_path:
            # load json file
            data = json.load(json_path)

            for element in data:
                caption = element['caption']
                video_name = element['video_name']
                # video_path = self.train_folder / f"{video_name}.mp4"
                # if not video_path.exists():
                #    continue
                captions.append(caption)
        
        caption_lengths = get_captions_length(captions, tokenizer)
        return caption_lengths

    def video_lengths(self) -> None:

        print('Analyzing video lengths of the dataset if there is a corresponding video...')
        video_lengths = []
        with open(self.default_train_annotations, 'r') as json_path:
            # load json file
            data = json.load(json_path)

            train_video_lengths = []
            val_video_lengths = []

            for video in data['videos']:
                video_id = video['id']
                start_time = video['start time']
                end_time = video['end time']
                video_length = round(end_time - start_time, 2)

                stat = {'video_id': video_id, 'video_length': video_length}
                if video['split'] == 'train':
                    train_video_lengths.append(stat)
                elif video['split'] == 'validate':
                    val_video_lengths.append(stat)

        
        with open(self.default_test_annotations, 'r') as json_path:
            # load json file
            data = json.load(json_path)

            test_video_lengths = []

            for video in data['videos']:
                video_id = video['id']
                start_time = video['start time']
                end_time = video['end time']
                video_length = end_time - start_time

                stat = {'video_id': video_id, 'video_length': video_length}
                test_video_lengths.append(stat)

        return train_video_lengths, val_video_lengths, test_video_lengths

    def check_video(self) -> None:

        train_videos = self.train_folder.glob('*.mp4')
        test_videos = self.test_folder.glob('*.mp4')
        for video in tqdm(train_videos):
            v, a, t = read_video(str(video))
            try:
                fps = t['audio_fps']
            except KeyError:
                print(f'{video} has no audio data in train folder.')

        # check test videos
        for video in tqdm(test_videos):
            v, a, t = read_video(str(video))
            try:
                fps = t['audio_fps']
            except KeyError:
                print(f'{video} has no audio data in test folder.')
