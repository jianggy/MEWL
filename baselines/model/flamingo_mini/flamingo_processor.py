from __future__ import annotations
from typing import List, Tuple
from PIL import Image

import torch
from transformers import AutoTokenizer


from .configuration_flamingo import FlamingoConfig


class FlamingoTokenizer:
    """ 
    FlamingoProcessor offers functions to preprocess the raw data (images and text).
    Wrapper around a transformer GPT-2 tokenizer and a clip processor.
    """
    
    # vision_processor: CLIPImageProcessor

    def __init__(
        self,
        config: FlamingoConfig,
        use_fast: bool = True,
        eoc_token: str = '<EOC>'
    ):
        """
        Args:
            config (FlamingoConfig): pass the same FlamingoConfig as used to initialize the FlamingoModel.
            use_fast (bool): whether to use the fast tokenizer implementations from huggingface.
            eoc_token (str): string representation of the token to add.
        """
        self.config = config
        self.eoc_token = eoc_token

        self.tokenizer = AutoTokenizer.from_pretrained(config.lm, use_fast=use_fast)
        self.tokenizer.add_bos_token = True
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_tokens(self.eoc_token)

        # find the start token for "<image>". " <" is 1279, "<" is 27
        # the encoded "<" token-id is different if there is a preceding whitespace.
        #        with ws    without
        # gpt-2:  1279         27
        # opt:   28696      51552
        self.leq_ids = [
            self.tokenizer.encode("<")[-1],
            self.tokenizer.encode(" <")[-1]
        ]

    def encode_text(
        self,
        text: str | List[str],
        device: torch.device | None = None,
        max_length=None,
        length=None,
        return_tensors='pt',
        return_attention_mask=True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if length is not None:
            result = self.tokenizer(
                text,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                padding='max_length',
                truncation=True,
                max_length=length)
        elif max_length is None:
            result = self.tokenizer(
                text,
                return_tensors=return_tensors, 
                padding=True)
        else:
            result = self.tokenizer(
                text,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                padding=True,
                truncation=True,
                max_length=max_length)
            
            
        media_locs = self.get_media_locations(result.input_ids)

        return result.input_ids.to(device), media_locs.to(device), result.attention_mask.to(device)
        
    def _remove_tags(self, text: str) -> str:
        for s in ('<image>', self.tokenizer.eos_token, self.eoc_token, self.tokenizer.pad_token):
            text = text.replace(s, '')
        return text.strip()
    
    def remove_tags(self, text: str | List[str]) -> str | List[str]:
        if isinstance(text, str):
            return self._remove_tags(text)
        else:
            return [self._remove_tags(t) for t in text]
    
    def get_media_locations(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.stack([(input_ids == leq_id) for leq_id in self.leq_ids]).sum(0)
    
    def __call__(
        self, 
        text: str | List[str] | None = None, 
        device: torch.device | None = None
    ):
        result = {}
        
        if text is not None:
            input_ids, media_locations, attention_mask = self.encode_text(text, device=device)
            result['input_ids'] = input_ids
            result['media_locations'] = media_locations
            result['attention_mask'] = attention_mask

        return result
