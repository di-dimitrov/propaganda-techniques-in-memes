import os
import json

import numpy as np
import torch

from PIL import Image

from mmf.common.registry import registry
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.general import get_mmf_root
from mmf.utils.text import VocabFromText, tokenize

def get_labels_vocab(data):
    all_labels = set([])
    for x in data:
        for l in x['labels']:
            if l not in all_labels:
                all_labels.add(l)
    label_vocab = {k: v for k, v in zip(all_labels, range(len(all_labels)))}
    return label_vocab

class PropagandaTask3FeaturesDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="propaganda", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)
        assert (
            self._use_features
        )
        
        self._data_dir = os.path.join(get_mmf_root(), config.data_dir)
        self._data_folder = self._data_dir
       
        if not os.path.exists(self._data_folder):
            raise RuntimeError(
                "Data folder {} for Propaganda is not present".format(self._data_folder)
            )
            
        if len(os.listdir(self._data_folder)) == 0:
            raise RuntimeError("Propaganda dataset folder is empty")
        self._load()
        
    def preprocess_sample_info(self, sample_info):
        image_path = sample_info["image"]
        feature_path = image_path.split(".")[0]
        sample_info["feature_path"] = f"{feature_path}.npy"
        return sample_info
        
    def _load(self):
        self.labels = get_labels_vocab(self.annotation_db)

    def __len__(self):
        return len(self.annotation_db)

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        sample_info = self.preprocess_sample_info(sample_info)
        
        current_sample = Sample()

        processed_text = self.text_processor({"text": sample_info["text"]})
        current_sample.text = processed_text["text"]
        
        if "input_ids" in processed_text:
            current_sample.update(processed_text)

        if "batch_2" in sample_info['id']:
            id = int(sample_info['id'].split("_batch_2")[0]) + 2000
        else:
            id = int(sample_info['id'])
        current_sample.id = torch.tensor(id, dtype=torch.int)

        features = self.features_db.get(sample_info)
        if hasattr(self, "transformer_bbox_processor"):
            features["image_info_0"] = self.transformer_bbox_processor(
                features["image_info_0"]
            )
        current_sample.update(features)

        label = torch.zeros(22)
        label[[self.labels[tgt] for tgt in sample_info["labels"]]] = 1
        current_sample.targets = label
        
        #current_sample.image = self.image_db[idx]["images"][0]

        return current_sample

class PropagandaTask3Dataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="propaganda", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)
        assert (
            self._use_images
        )
        
        self._data_dir = os.path.join(get_mmf_root(), config.data_dir)
        self._data_folder = self._data_dir
       
        if not os.path.exists(self._data_folder):
            raise RuntimeError(
                "Data folder {} for Propaganda is not present".format(self._data_folder)
            )
            
        if len(os.listdir(self._data_folder)) == 0:
            raise RuntimeError("Propaganda dataset folder is empty")
        self._load()
        
    def init_processors(self):
        super().init_processors()
        self.image_db.transform = self.image_processor
        
    def _load(self):
        #self.image_path = os.path.join(self._data_folder,"propaganda/defaults/images/")
        #with open(
        #    os.path.join(
        #        #FIX TO MATCH MY DIRECTORY AND FILES
        #        self._data_folder,
        #        "propaganda",
        #        "defaults",
        #        "annotations",
        #        "train.json",
        #    ),encoding="utf-8"
        #) as f:
        #    self.data = json.load(f)
        self.labels = get_labels_vocab(self.annotation_db)

    def __len__(self):
        return len(self.annotation_db)

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        #data = self.data[idx]
        current_sample = Sample()

        processed_text = self.text_processor({"text": sample_info["text"]})
        current_sample.text = processed_text["text"]
        
        if "input_ids" in processed_text:
            current_sample.update(processed_text)
            
        if "batch_2" in sample_info['id']:
            id = int(sample_info['id'].split("_batch_2")[0]) + 2000
        else:
            id = int(sample_info['id'])
        current_sample.id = torch.tensor(id, dtype=torch.int)

        label = torch.zeros(22)
        label[[self.labels[tgt] for tgt in sample_info["labels"]]] = 1
        current_sample.targets = label

        #image_path = os.path.join(self.image_path, data["image"])
        #image = np.true_divide(Image.open(image_path).convert("RGB"), 255)
        #image = image.astype(np.float32)

        #current_sample.image = torch.from_numpy(image.transpose(2, 0, 1))
        
        current_sample.image = self.image_db[idx]["images"][0]

        return current_sample
