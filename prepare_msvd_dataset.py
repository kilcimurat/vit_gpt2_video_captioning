from pathlib import Path
import pathlib
import os
from typing import List, Tuple, Optional
from torchvision.datasets.utils import download_url, extract_archive
import pytube
from pytube import YouTube
from urllib.request import HTTPError
from video_utils import clip_video
import json
import shutil
import av
from urllib.error import URLError
from socket import gaierror
from http.client import IncompleteRead
from tqdm import tqdm

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
def load_list(path: pathlib.Path, video_folder: pathlib.Path, update_annotations, save_path) -> Tuple[Tuple[List[str], List[str], List[str]], Tuple[List[str], List[str], List[str]]]:
    # Initialize video_paths list.
    train_video_paths = []
    test_video_paths = []
    # Initialize ids list.
    train_video_ids = []
    test_video_ids = []
    # Initialize train captions list.
    train_captions = []
    test_captions = []

    with open(str(path)) as f:
        annotations = f.readlines()
        annotations = annotations[7:]
 
    downloaded_video_paths = sorted(video_folder.glob('*.*'))
    file_map = {path.stem: path.name for path in downloaded_video_paths}
    ids = sorted(file_map.keys())

    split_index = int(len(ids) * 0.7)
    train_set_ids = ids[:split_index]
    test_set_ids = ids[split_index:]

    for vid in tqdm(train_set_ids):
        filename = file_map.get(vid)
        if not filename:
            continue
        video_path = video_folder / filename
        for ann in annotations:
            if vid == ann.split()[0]:
                train_video_paths.append(video_path)
                train_video_ids.append(vid)
                caption = ' '.join(ann.split()[1:])
                train_captions.append(caption)

    test_annotations = []
    for vid in tqdm(test_set_ids):
        filename = file_map.get(vid)
        if not filename:
            continue
        video_path = video_folder / filename
        for ann in annotations:
            if vid == ann.split()[0]:
                test_video_paths.append(video_path)
                test_video_ids.append(vid)
                caption = ' '.join(ann.split()[1:])
                test_captions.append(caption)

                test_data = {'video_id': vid, 'video_name': video_path.name, 'caption': caption}
                test_annotations.append(test_data)

    if update_annotations:
        with open(save_path, 'w') as f:
            json.dump(test_annotations, f)




    
    return train_video_paths, train_captions, train_video_ids, test_video_paths, test_captions, test_video_ids 

class MSVDDataset():
    '''
        Utilities for video captioning dataset of MSVD

        Initialize:
        # dt = MSVDDataset()

        Download MSVD Dataset:
        # dt.download_dataset()

        Load captions, paths, times and ids:
        # train_data, test_data = dt.load_data()

        # paths, captions, ids, start_times, end_times = zip(*train_data)
        In test data captions are not important therefore only one corresponding caption for a video added.
        # paths, captions, ids, start_times, end_times = zip(*test_data)

    '''
    def __init__(self, root_folder: Optional[pathlib.Path] = None) -> None:

        # name of the dataset.
        self.name = "MSVD"

        # url links for the dataset annotations in zip format.
        # self.videos = ["https://drive.google.com/file/d/11IBEBbAOHFq0rIiWBlaFvqMeqErnGzNG", "YouTubeClips.tar"]

        env = os.getenv

        # url links for the dataset annotations.
        self.annotation_filename = "AllVideoDescriptions.txt"
        annotation_override = env('MSVD_ANN_URL')
        self.annotation_sources = [
            (annotation_override, self.annotation_filename),
            ("https://www.cs.utexas.edu/users/ml/clamp/videoDescription/AllVideoDescriptions.txt", self.annotation_filename),
            ("http://www.cs.utexas.edu/users/ml/clamp/videoDescription/AllVideoDescriptions.txt", self.annotation_filename),
        ]

        # direct archive containing trimmed MSVD clips
        self.video_archive_filename = "YouTubeClips.tar"
        archive_override = env('MSVD_ARCHIVE_URL')
        self.video_archive_sources = [
            (archive_override, self.video_archive_filename),
            ("https://huggingface.co/datasets/OneDiffusion/MSVD/resolve/main/YouTubeClips.tar?download=1", self.video_archive_filename),
            ("https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar", self.video_archive_filename),
            ("http://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar", self.video_archive_filename),
        ]

        # Project root Path
        default_root = Path(os.getenv('MSVD_ROOT', '/Volumes/KINGSTON/dataset'))
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

        # train visual features Path
        self.train_visual_features_folder = self.dataset_folder / "features_visual_train"

        # train audial features Path
        self.train_audial_features_folder = self.dataset_folder / "features_audial_train"

        # test features Path
        self.test_visual_features_folder = self.dataset_folder / "features_visual_test"

        # test features Path
        self.test_audial_features_folder = self.dataset_folder / "features_audial_test"

        # description path
        self.annotations = self.dataset_folder / "AllVideoDescriptions.txt"
        # train captions path
        self.train_annotations = self.dataset_folder / "train_annotations.json"
        self.val_annotations = None

        # test captions path
        self.test_annotations = self.dataset_folder / "test_annotations.json"

        self.max_visual_frame_length = 1801
        self.max_visual_seq_length = 240
        self.max_audial_frame_length = 2648064
        self.max_audial_seq_length = 517
        self.min_frames_audio = 83968

        self.min_caption_length = 3
        self.max_caption_length = 12

    def download_annotations(self) -> None:
        '''
            Download the dataset's train videos, test videos and their annotations into the MSRVTT folder.
        '''

        target = self.dataset_folder / self.annotation_filename
        if target.exists():
            print(f"{target.name} already exists.")
            return

        if not self.dataset_folder.exists():
            self.dataset_folder.mkdir(parents=True, exist_ok=True)

        last_error = None
        for url, filename in self.annotation_sources:
            if not url:
                continue
            try:
                print(f"Downloading {filename} from {url}...")
                download_url(url=url, root=str(self.dataset_folder), filename=filename)
                return
            except Exception as exc:
                print(f"Failed to download {filename} from {url} ({exc}).")
                last_error = exc

        raise RuntimeError("Unable to download MSVD annotations from any configured source.") from last_error

    def download_video_archive(self) -> None:
        """Download the official MSVD clip archive if local videos are missing."""
        if self.train_folder.exists() and any(self.train_folder.glob('*.*')):
            return
        archive_path = self.dataset_folder / self.video_archive_filename
        if not self.dataset_folder.exists():
            self.dataset_folder.mkdir(parents=True, exist_ok=True)

        if not archive_path.exists():
            last_error = None
            for url, filename in self.video_archive_sources:
                if not url:
                    continue
                try:
                    print(f"Downloading {filename} from {url}...")
                    download_url(url=url, root=str(self.dataset_folder), filename=filename)
                    archive_path = self.dataset_folder / filename
                    if filename != self.video_archive_filename:
                        target_archive = self.dataset_folder / self.video_archive_filename
                        if target_archive.exists():
                            target_archive.unlink()
                        archive_path.rename(target_archive)
                        archive_path = target_archive
                    break
                except Exception as exc:
                    print(f"Failed to download {filename} from {url} ({exc}).")
                    last_error = exc
            else:
                raise RuntimeError("Unable to download MSVD video archive from any configured source.") from last_error

        print(f"Extracting {self.video_archive_filename}...")
        extract_archive(str(archive_path), str(self.dataset_folder))
        archive_root = self.dataset_folder / 'YouTubeClips'
        if archive_root.exists():
            self.train_folder.mkdir(parents=True, exist_ok=True)
            for video_file in archive_root.glob('*.*'):
                target = self.train_folder / video_file.name
                if target.exists():
                    continue
                shutil.move(str(video_file), str(target))
            try:
                archive_root.rmdir()
            except OSError:
                pass
        else:
            print('Archive extracted but expected YouTubeClips folder not found; keeping existing layout.')

    def download_videos(self) -> None:
        '''
            Read txt file and download videos from youtube
        '''
        annotation_path = self.dataset_folder / self.annotation_filename
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        annotations = lines[7:]
        video_ids = []
        for annotation in annotations:
            video_id = annotation.split(" ")[0]
            video_ids.append(video_id)
        video_ids = list(set(video_ids))
        
        status = []

        for video_id in video_ids:
            start_time = float(video_id[12:].split("_")[0])
            end_time = float(video_id[12:].split("_")[1])
            youtube_id = video_id[:11]

            stat, is_available = self.download_video(
                youtube_id=youtube_id,
                clip_id=video_id,
                start_time=start_time,
                end_time=end_time,
                download_folder=self.train_folder,
            )
            status.append({
                'video_id': youtube_id,
                'clip_id': video_id,
                'start_time': start_time,
                'end_time': end_time,
                'status': stat,
                'is_available': is_available,
            })
            
        # write status to json file.
        with open(str(self.dataset_folder / 'download_status.json'), 'w') as f:
            json.dump(status, f)

    # download youtube videos start time to end time from id.
    def download_video(self, youtube_id: str, clip_id: str, start_time: float, end_time: float, download_folder: Path) -> Tuple[str, bool]:

        '''
            Download youtube videos start time to end time from id.
        '''

        # youtube video url.
        url = f"https://www.youtube.com/watch?v={youtube_id}"

        yt = YouTube(url)
        try:
            download_folder.mkdir(parents=True, exist_ok=True)
            temp_basename = f"{clip_id}_full"
            temp_video_path = download_folder / f"{temp_basename}.mp4"
            clipped_video_path = download_folder / f"{clip_id}.mp4"
            if clipped_video_path.exists():
                print(f"Clip already exists: {clipped_video_path.name}")
                stat = "Available"
                return stat, True

            stream = yt.streams.filter(file_extension="mp4", resolution="360p").first()
            if stream is None:
                stream = yt.streams.filter(file_extension="mp4").order_by('resolution').desc().first()
            if stream is None:
                print("No MP4 stream available: " + youtube_id)
                stat = "No Stream"
                return stat, False

            stream.download(output_path=str(download_folder), filename=temp_basename)
            clip_video(video_path=temp_video_path, start_time=start_time, end_time=end_time, output_path=str(clipped_video_path))
            if temp_video_path.exists():
                temp_video_path.unlink()
            print("Downloaded: " + clip_id)
            stat = "Available"
            return stat, True
        except pytube.exceptions.VideoUnavailable:
            print("Video Unavailable: " + youtube_id)
            stat = "Unavailable"
            return stat, False
        except KeyError:
            print("Key Error: " + youtube_id)
            stat = "Key Error"
            return stat, False
        except HTTPError:
            print("HTTP Error: " + youtube_id)
            stat = "HTTP Error"
            return stat, False
        except IncompleteRead:
            print("Incomplete Read: " + youtube_id)
            stat = "Incomplete Read Error"
            if temp_video_path.exists():
                temp_video_path.unlink()
            return stat, False
        except av.error.ValueError:
            print("Value Error: " + youtube_id)
            stat = "Value Error"
            return stat, False
        

    def download_dataset(self) -> None:
        self.download_annotations()
        try:
            self.download_video_archive()
        except Exception as exc:
            print(f"Archive download failed ({exc}); falling back to YouTube downloads.")
            self.download_videos()
        else:
            if not any(self.train_folder.glob('*.*')):
                print("Archive extraction produced no videos; falling back to YouTube downloads.")
                self.download_videos()
        self.update_annotations()


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
            Load the MSVD captions and their corresponding video ids.
            paths, captions, ids, start_times, end_times
        '''

        train_names, train_captions, train_ids = self.load_annotations(self.train_annotations)
        train_data = zip(train_names, train_captions, train_ids)
        val_data = None
        test_names, test_captions, test_ids = self.load_annotations(self.test_annotations)
        test_data = zip(test_names, test_captions, test_ids)

        return train_data, val_data, test_data

    def update_annotations(self):
        train_video_paths = []
        test_video_paths = []
        train_video_ids = []
        test_video_ids = []
        train_captions = []
        test_captions = []

        with open(self.annotations) as f:
            raw_lines = f.readlines()
            annotations = raw_lines[7:]

        downloaded_video_paths = sorted(self.train_folder.glob('*.*'))
        file_map = {path.stem: path.name for path in downloaded_video_paths}
        ids = sorted(file_map.keys())
        if not ids:
            print('No downloaded MSVD videos found; skip annotation update.')
            return

        split_index = int(len(ids) * 0.7)
        train_set_ids = ids[:split_index]
        test_set_ids = ids[split_index:]

        annotation_lookup = {}
        for ann in annotations:
            parts = ann.strip().split()
            if not parts:
                continue
            video_id = parts[0]
            caption = ' '.join(parts[1:])
            if not caption:
                continue
            annotation_lookup.setdefault(video_id, []).append(caption)

        train_annotations = []
        test_annotations = []

        for video_id in tqdm(train_set_ids):
            filename = file_map.get(video_id)
            if not filename:
                continue
            captions = annotation_lookup.get(video_id, [])
            if not captions:
                continue
            video_path = self.train_folder / filename
            for caption in captions:
                train_video_paths.append(video_path)
                train_video_ids.append(video_id)
                train_captions.append(caption)
                train_annotations.append({'video_id': video_id, 'video_name': video_path.name, 'caption': caption})

        for video_id in tqdm(test_set_ids):
            filename = file_map.get(video_id)
            if not filename:
                continue
            captions = annotation_lookup.get(video_id, [])
            if not captions:
                continue
            video_path = self.train_folder / filename
            for caption in captions:
                test_video_paths.append(video_path)
                test_video_ids.append(video_id)
                test_captions.append(caption)
                test_annotations.append({'video_id': video_id, 'video_name': video_path.name, 'caption': caption})

        with open(self.train_annotations, 'w') as f:
            json.dump(train_annotations, f)

        with open(self.test_annotations, 'w') as f:
            json.dump(test_annotations, f)
