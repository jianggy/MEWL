import torch
import pathlib
import os
import torch
import torchvision
import numpy as np
import clip

from torch import nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import json
import pytorch_lightning as pl

from model.flamingo_mini import FlamingoConfig
from model.consts import task_names
import datasets

from model.consts import task_names
# import datasets
from copy import deepcopy

BATCH_SIZE = 8
NUM_FRAMES = 25
NUM_OBJECTS = 8
QUESTION_VOCAB_SIZE = 82
ANSWER_VOCAB_SIZE = 22

MAX_QUESTION_LENGTH = 20
MAX_CHOICE_LENGTH = 7

MAX_SEQ_LEN = 2048

def _split_string(s):
    """Splits string to words and standardize alphabet."""
    return s.lower().replace("?", "").split()


def _pad(array, length):
    """Pad an array to desired length."""
    if length - array.shape[0] < 0:
        return array[: length]
    return np.pad(array, [(0, length - array.shape[0])] + [(0, 0) for _ in range(len(array.shape) - 1)], mode="constant")


def encode_sentence_new(token_map, sentence, pad_length):
    """Encode CLEVRER question/choice sentences as sequence of token ids."""
    ret = np.array([token_map.get(w, 0) for w in _split_string(sentence)], np.int32)
    return _pad(ret, pad_length)

def encode_scene(token_map, scene):
    """Encode CLEVRER choices."""
    # scene 11 text segments
    token_map_new = deepcopy(token_map)
    for s in scene:
        for w in _split_string(s):
            if w not in token_map_new:
                token_map_new[w] = len(token_map_new) + 1
    # print(token_map_new)
    arrays = [encode_sentence_new(token_map_new, s, MAX_CHOICE_LENGTH) for s in scene]
    ret = np.stack(arrays, axis=0)
    return ret


class MEWLDatasetForAloe(Dataset):
    def __init__(self, dataset_path, split, task_name='*', context_length=77):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        self.split = split
        # self.transform = self._clip_transform()
        self.context_length = context_length

        assert split in ['train', 'val', 'test']
        self.episodes = list(self.dataset_path.glob(f'{split}/{task_name}/*'))
        self.task_names = [
            "composite",
            "material",
            "object",
            "bootstrap",
            "shape",
            "color",
            "number",
            "pragmatic",
            "relation",
        ]
        self.task_name2idx = {task_name: idx for idx, task_name in enumerate(self.task_names)}

        with open("vocab_small.json", "rb") as f:
            self.token_map = json.load(f)["vocab"]
        # token_map = {}

    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    def __len__(self):
        return len(list(self.episodes))

    def __getitem__(self, index):
        episode_path = list(self.episodes)[index]
        json_path = episode_path / 'info.json'
        task_idx = self.task_name2idx[episode_path.parent.name]

        with open(json_path) as f:
            info = json.load(f)

        # images = []
        texts = []
        # todo: may require some randomization to ensure the randomness of placeholder
        for context in info['context_panels']:
            context_caption = context['name']
            # image = Image.open(episode_path / f"{context_caption}.png")
            # images.append(image)

            text = context_caption.split(' ')
            if text[-1][0] == '#':
                text = text[:-1]

            texts.append(' '.join(text))
        
        # texts = ' '.join(texts)

        # print(texts)

        # images.append(Image.open(episode_path / "question.png"))
        
        choices = info['question']['choices']
        answer = info['question']['answer']

        answer_idx = choices.index(answer)
        texts += choices

        # print(texts) #  batch * 11 * words
        texts = encode_scene(self.token_map, texts)

        # images = torch.stack([self.transform(image) for image in images])
        images = np.load(episode_path / "latent.npy")

        return images, texts, answer_idx, task_idx

class MEWLAloeData(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size=64, num_workers=8, context_length=77, **kwargs):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = MEWLDatasetForAloe(self.dataset_path, 'train', context_length=context_length)
        self.val_dataset = MEWLDatasetForAloe(self.dataset_path, 'val', context_length=context_length)
        self.test_dataset = MEWLDatasetForAloe(self.dataset_path, 'test', context_length=context_length)

    def train_dataloader(self):
        # distributed
        if self.trainer.num_devices > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset)
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=train_sampler)  
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        # distributed
        if self.trainer.num_devices > 1:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_dataset)
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=val_sampler)
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)        

    def test_dataloader(self):
        # distributed
        if self.trainer.num_devices > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                self.test_dataset)
            return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=test_sampler)
        else:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)   

class MEWLDatasetForCLIP(Dataset):
    def __init__(self, dataset_path, split, task_name='*', context_length=77):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        self.split = split
        self.transform = self._clip_transform()
        self.context_length = context_length

        assert split in ['train', 'val', 'test']
        self.episodes = list(self.dataset_path.glob(f'{split}/{task_name}/*'))
        self.task_names = [
            "composite",
            "material",
            "object",
            "bootstrap",
            "shape",
            "color",
            "number",
            "pragmatic",
            "relation",
        ]
        self.task_name2idx = {task_name: idx for idx, task_name in enumerate(self.task_names)}

    def _clip_transform(self, n_px=224):
        return Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            self._convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    def _clip_tokenize(self, text):
        return clip.tokenize(text, context_length=self.context_length)

    def __len__(self):
        return len(list(self.episodes))

    def __getitem__(self, index):
        episode_path = list(self.episodes)[index]
        json_path = episode_path / 'info.json'
        task_idx = self.task_name2idx[episode_path.parent.name]

        with open(json_path) as f:
            info = json.load(f)

        images = []
        texts = []

        for context in info['context_panels']:
            context_caption = context['name']
            image = Image.open(episode_path / f"{context_caption}.png")
            images.append(image)

            text = context_caption.split(' ')
            if text[-1][0] == '#':
                text = text[:-1]

            texts.append(' '.join(text))

        images.append(Image.open(episode_path / "question.png"))
        choices = info['question']['choices']
        answer = info['question']['answer']

        answer_idx = choices.index(answer)
        texts += choices

        images = torch.stack([self.transform(image) for image in images])
        texts = self._clip_tokenize(texts)

        return images, texts, answer_idx, task_idx


class MEWLDatasetForFlamingo(Dataset):
    def __init__(self, dataset_path, split, task_name='*'):
        self.dataset_path = pathlib.Path(dataset_path)
        self.split = split
        self.transform = self._clip_transform()

        assert split in ['train', 'val', 'test']
        self.episodes = list(self.dataset_path.glob(f'{split}/{task_name}/*'))
        self.task_names = [
            "composite",
            "material",
            "object",
            "bootstrap",
            "shape",
            "color",
            "number",
            "pragmatic",
            "relation",
        ]
        self.task_name2idx = {task_name: idx for idx, task_name in enumerate(self.task_names)}
        self.prefix = '<image>'
        self.eoc = '<EOC>'
    

    def _clip_transform(self, n_px=224):
        return Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            self._convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    def __len__(self):
        return len(list(self.episodes))

    def __getitem__(self, index):
        episode_path = list(self.episodes)[index]
        json_path = episode_path / 'info.json'
        task_idx = self.task_name2idx[episode_path.parent.name]

        with open(json_path) as f:
            info = json.load(f)

        images = []
        texts = []

        text_prefix = ""

        for context in info['context_panels']:
            context_caption = context['name']
            image = Image.open(episode_path / f"{context_caption}.png")
            images.append(image)

            text = context_caption.split(' ')
            if text[-1][0] == '#':
                text = text[:-1]

            text_prefix += self.prefix + ' ' + ' '.join(text) + ' ' + self.eoc + ' '
        
        images.append(Image.open(episode_path / "question.png"))

        choices = info['question']['choices']
        text_choices = [text_prefix + self.prefix + ' ' + choice + ' ' + self.eoc for choice in choices]

        answer = info['question']['answer']
        answer_idx = choices.index(answer)

        images = torch.stack([self.transform(image) for image in images])

        return images, text_choices, answer_idx


class MEWLCLIPData(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size=32, num_workers=4, context_length=77, **kwargs):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = MEWLDatasetForCLIP(self.dataset_path, 'train', context_length=context_length)
        self.val_dataset = MEWLDatasetForCLIP(self.dataset_path, 'val', context_length=context_length)
        self.test_dataset = MEWLDatasetForCLIP(self.dataset_path, 'test', context_length=context_length)

    def train_dataloader(self):
        # distributed
        if self.trainer.num_devices > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset)
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=train_sampler)  
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        # distributed
        if self.trainer.num_devices > 1:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_dataset)
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=val_sampler)
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)        

    def test_dataloader(self):
        # distributed
        if self.trainer.num_devices > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                self.test_dataset)
            return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=test_sampler)
        else:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)     
            

def get_MEWL_huggingface(path, taskname=None):
    agg_dataset = {'train': [], 'val': [], 'test': []}
    
    if taskname is None:
        taskname = task_names

    if not isinstance(taskname, list):
        taskname = [taskname]
    
    for task in taskname:
        json_path = os.path.join(path, f"{task}.json")
        with open(json_path) as f:
            data = json.load(f)

        for split in ['train', 'val', 'test']:
            agg_dataset[split] += data[split]

    for split in ['train', 'val', 'test']:
        agg_dataset[split] = datasets.Dataset.from_list(agg_dataset[split])

    return datasets.dataset_dict.DatasetDict(agg_dataset)
