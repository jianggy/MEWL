from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import contextlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from transformers import PreTrainedModel
from transformers.models.clip.modeling_clip import CLIPVisionModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    MultipleChoiceModelOutput,
)

from .configuration_flamingo import FlamingoConfig
from .gated_cross_attention import ModifiedLMBlock
from .perceiver_resampler import PerceiverResampler


@contextlib.contextmanager
def suppress_model_loading_warnings(suppress: bool = True):
    if suppress:
        logger = logging.getLogger('transformers.modeling_utils')
        level = logger.level
        logger.setLevel(logging.CRITICAL)
        yield
        logger.setLevel(level)
    else:
        yield




class FlamingoBaseModel(ABC, PreTrainedModel):
    """ 
    abstract class, which is inherited by FlamingoGPT2 and FlamingoOPT.
    This class provides the core functionalities of Flamingo: the forward() function,
    setting up the resampler and hijacking the LM layers with GatedXAttn layers.
    """

    config: FlamingoConfig
    vision_encoder: CLIPVisionModel
    resampler: PerceiverResampler
    lm: PreTrainedModel

    config_class = FlamingoConfig

    def __init__(self, config: FlamingoConfig, suppress_warnings=True):
        assert isinstance(config, FlamingoConfig)
        super().__init__(config)
        
        with suppress_model_loading_warnings(suppress_warnings):
            self.vision_encoder = CLIPVisionModel.from_pretrained(config.clip_model_type) # type: ignore

        self.resampler = PerceiverResampler(
            dim=config.dim_visual,
            depth=config.resampler_depth,
            dim_head=config.resampler_dim_head,
            heads=config.resampler_heads,
            num_latents=config.resampler_num_latents,
            num_time_embeds=config.resampler_num_time_embeds,
            ff_mult=config.resampler_ff_mult,
            act=config.resampler_act
        )

    def _init_layers(self, lm_layers: nn.ModuleList):
        """ 
        call during init of the subclass.
        careful, this method will modify the LM layers!
        """
        for i, lm_layer in enumerate(lm_layers):
            if i % self.config.xattn_every != 0: 
                continue

            lm_layers[i] = ModifiedLMBlock(
                lm_layer,
                dim=self.config.dim,
                dim_visual=self.config.dim_visual,
                dim_head=self.config.xattn_dim_head,
                heads=self.config.xattn_heads,
                ff_mult=self.config.xattn_ff_mult,
                act=self.config.xattn_act,
                n_visual=self.config.resampler_num_latents
            )
            
    @abstractmethod
    def get_modified_layers(self) -> List[ModifiedLMBlock]:
        raise NotImplementedError
            
    def freeze_vm(self):
        """freeze vision model """
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def freeze_lm(self):
        """ freeze weights of the language model.

        (!) does not freeze token embedding matrix and gated xattn layers
        """

        for param in self.lm.parameters():
            param.requires_grad = False

        # lm_head shares weights with the embeddings so no need to unfreeze that as well
        self.lm.get_input_embeddings().weight.requires_grad = True

        for xattn in self.get_modified_layers():
            for param in xattn.xattn_block.parameters():
                param.requires_grad = True

    def unfreeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = True

    def state_dict_trainable(self) -> Dict[str, torch.Tensor]:
        """ include weights in the state dict if they have requires_grad = True"""

        trainable_param_names = [
            w for w, t in self.named_parameters() if t.requires_grad]
        return {k: v for k, v in self.state_dict().items() if k in trainable_param_names}

    def parameters_trainable(self):
        """Access the trainable parameters, e.g. useful for the optimizer and gradient clipping. 

        example: optimizer = AdamW(model.parameters_trainable(), lr=args.lr)
        make sure to call freeze_lm() first! 
        """
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def encode_resample_visuals(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pass pixel values through vision encoder and perceiver resampler.

        Args:
            pixel_values (torch.Tensor): accepted shapes:
                (N c h w)       one batch, multiple images
                (b N c h w)     multiple batches, multiple images
                (b N T c h w)   multiple batches, multiple images, multiple frames

        Returns:
            (torch.Tensor): shape (b N q d)
        """

        if pixel_values.ndim == 4:            
            # (N c h w)
            b, N, T = 1, pixel_values.size(0), 1
        
        elif pixel_values.ndim == 5:       
            # (b N c h w)
            b, N, T = *pixel_values.shape[:2], 1
            pixel_values = rearrange(pixel_values, 'b N c h w -> (b N) c h w')

        elif pixel_values.ndim == 6:         
            # (b N T c h w) -> (b N T v d)
            b, N, T = pixel_values.shape[:3]
            pixel_values = rearrange(pixel_values, 'b N T c h w -> (b N T) c h w')
        else:
            raise ValueError('pixel_values must have ndim 5 or 6!')

        with torch.no_grad():
            visual_features = self.vision_encoder(pixel_values).last_hidden_state         # (b N T) v d

        # perceiver resampler
        # (only need to do if kv of the xattn layers were not calculated yet.)
        # resample visual features ((b N T) v d) -> (b N T q d)
        visual_features = rearrange(visual_features, '(b N T) v d -> (b N) T v d', b=b, N=N, T=T)
        visual_features = self.resampler(visual_features)

        # T is gone at this point
        visual_features = rearrange(visual_features, '(b N) q d -> b N q d', b=b, N=N)
        
        return visual_features
        
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        media_locations: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        visual_features: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: tuple | None = None,
        return_dict: bool = True,
        loss_reduction: str = 'mean',
        **kwargs
    ) -> BaseModelOutputWithPast:
        """Flamingo forward pass

        Most of the parameters are inspired by huggingface language model implementations, so this doc may be informative:
        https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Model.forward

        Args:
            input_ids (Tensor | None):         shape (n_batch, n_tokens). the tokenized input text
            attention_mask (Tensor | None):    shape (n_batch, n_tokens). 
                Mask as produced by the tokenizer. Required when a batch of input strings are tokenized and thus padded at the end.
                Then this will indicate the locations of 'real' tokens vs. the location of 'pad' tokens.
            media_locations (Tensor | None):   shape (n_batch, n_tokens).
                indicates the locations of the starts of the <image> tags beginning, i.e. the location of the token representing '<'
            pixel_values (Tensor | None):    shape (b N T c h w). Optional.
            visual_features (Tensor | None):         shape (b N q d). Optional.
                If pixel_values already have been passed through encode_resample_visuals(), 
                you can pass the resampled visual embeddings via this parameter.
                If provided, pixel_values will be ignored
            head_mask (Tensor | None): TODO
            inputs_embeds (Tensor | None): TODO
            use_cache (bool): whether to return the inner keys and values. Used to speed up text generation at inference. defaults to False
            past_key_values (tuple): tuple of past_key_values of (1) the xattn layers (2) the language model
            return_dict (bool): Whether to return a dictionary. Right now, only dicts are supported, so this must be set to True. Defaults to True.
            labels (Tensor): 
                It is possible to pass the exact value as input_ids also as labels. If present, the output will contain a CE loss of the next token prediction.
                optional, defaults to None
            **kwargs

        Returns:
            (SequenceClassifierOutputWithPast): an object containing all the useful stuff. Refer to hf documentation.

        """

        # sanity check
        assert return_dict, "can only use return_dict=True at the moment!"
        assert (input_ids is None) != (inputs_embeds is None), "you must pass either input_ids or inputs_embeds!"

        # find the input shape
        batch_size, seq_length = input_ids.shape[:2] if input_ids is not None else inputs_embeds.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        xattn_past_key_values = None if past_key_values is None else past_key_values[0]
        lm_past_key_values = None if past_key_values is None else past_key_values[1]
        
        if visual_features is None:
            if xattn_past_key_values is None and pixel_values is not None:
                # extract from pixels
                assert pixel_values.size(0) == batch_size, \
                    "pixel_values must have the same batch size as the textual input!"
                
                visual_features = self.encode_resample_visuals(pixel_values)
                
            else:
                # we don't need visual_features is past is defined.
                # use dummy values, since are only required for the shape
                # visual_embedings shape (b N q d)
                visual_features = torch.zeros(
                    (batch_size, 1, self.config.resampler_num_latents, self.config.dim_visual),
                    dtype=torch.float32,
                    device=device
                ) 

        if media_locations is None:
            media_locations = torch.zeros(size=(batch_size, seq_length), dtype=torch.int, device=device)

        # condition xattn layers
        for i, xattn in enumerate(self.get_modified_layers()):
            layer_past = None if xattn_past_key_values is None else xattn_past_key_values[i]
            xattn.condition(visual_features, media_locations, layer_past)

        # pass through LM
        out: BaseModelOutputWithPast = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=lm_past_key_values,
            return_dict=True,
            **kwargs
        )

        return out

class FlamingoLM(FlamingoBaseModel):
    config: FlamingoConfig
    config_class = FlamingoConfig

    def __init__(self, config: FlamingoConfig):
        from transformers import OPTModel
        super().__init__(config)

        base_lm: OPTModel = OPTModel.from_pretrained(config.lm)  # type: ignore

        assert self.config.dim == base_lm.config.hidden_size, \
            f"specified {self.config.dim=} in FlamingoConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"

        base_lm.resize_token_embeddings(base_lm.config.vocab_size + 1)
        self.lm: OPTModel = base_lm
        self._init_layers(self.lm.decoder.layers)
        
    def get_modified_layers(self):
        if self.config.xattn_every == 1:
            return self.lm.decoder.layers
        return filter(lambda layer: isinstance(layer, ModifiedLMBlock), self.lm.decoder.layers)


class FlamingoModelForMultipleChoice(PreTrainedModel):
    """wrapper class for a FlamingoBase decending model (FlamingoGPT2 or FlamingoOPT)

    A generic flamingo interface that is independent of the underlying LM. Most of the methods are just forwarding to the actual model.
    This class implements prepare_inputs_for_generation() and reorder_cache(), which are required to utilize hf text generation methods.
    It also has a generate_captions() utility that can be used to create a caption for an image.
    """
    config: FlamingoConfig
    config_class = FlamingoConfig

    # key = prefix of an existing pretrained huggingface transformer language model
    # value = Flamingo class for the respective language model
    
    _keys_to_ignore_on_load_missing = [r"flamingo.vision_encoder"]

    def __init__(self, config: FlamingoConfig, model_class: type | None = None):
        """constructor.

        Args:
            config (FlamingoConfig): 
                config for the flamingo model.
            model_class (Optional[type], optional): 
                optionally use a custom class that inherits FlamingoBaseModel. 
                If none, it will choose FlamingoGPT2 or FlamingoOPT based on the FlamingoConfig. Defaults to None.
        """
        super().__init__(config)

        self.flamingo: FlamingoBaseModel = FlamingoLM(config)
        self.flamingo.lm.decoder.project_out = None
        
        if config.freeze_language_model:
            self.freeze_lm()

        if config.freeze_vision_model:
            self.freeze_vm()

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.dim, 1)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def parameters_trainable(self):
        """Access the trainable parameters, e.g. useful for the optimizer and gradient clipping. 

        example: optimizer = AdamW(model.parameters_trainable(), lr=args.lr)
        make sure to call freeze_lm() first! 
        """
        return self.flamingo.parameters_trainable()

    def freeze_vm(self):
        self.flamingo.freeze_vm()

    def freeze_lm(self):
        self.flamingo.freeze_lm()

    def unfreeze_lm(self):
        self.flamingo.unfreeze_lm()

    def state_dict_trainable(self):
        return self.flamingo.state_dict_trainable()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        media_locations: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None, # N T c h w
        visual_features: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: tuple | None = None,
        return_dict: bool = True,
        labels: torch.Tensor | None = None,
        loss_reduction: str = 'mean',
        **kwargs
    ) -> MultipleChoiceModelOutput:

        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = rearrange(input_ids, 'b n c -> (b n) c') if input_ids is not None else None
        attention_mask = rearrange(attention_mask, 'b n c -> (b n) c') if attention_mask is not None else None
        media_locations = rearrange(media_locations, 'b n c -> (b n) c') if media_locations is not None else None
        head_mask = rearrange(head_mask, 'b n c -> (b n) c') if head_mask is not None else None
        inputs_embeds = rearrange(inputs_embeds, 'b n c -> (b n) c') if inputs_embeds is not None else None
        past_key_values = rearrange(past_key_values, 'b n c -> (b n) c') if past_key_values is not None else None

        
        if visual_features is None:
            visual_features = self.flamingo.encode_resample_visuals(pixel_values)
            visual_features = repeat(visual_features, 'b t q d -> (b n) t q d', n=num_choices)
        else:
            visual_features = rearrange(visual_features, 'b n c -> (b n) c')

        out = self.flamingo(
            input_ids=input_ids,
            attention_mask=attention_mask,
            media_locations=media_locations,
            pixel_values=pixel_values,
            visual_features=visual_features,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            loss_reduction=loss_reduction,
            **kwargs
        )

        logits = out[0]

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        pooled_output = self.dropout(pooled_logits)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        
        if not return_dict:
            output = (reshaped_logits,) + out[1:]
            return ((loss,) + output) if loss is not None else output
        
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )

    def _reorder_cache(self, past, beam_idx):
        """ hf specific function. Overridden from PreTrainedModel.

        this is required for beam search in combination with use_cache.

        Args: 
            past is a tuple of past_key_values of the xattn layers, and of the LM layers.
            beam_idx: index of the beam
        """
        xattn_past, lm_past = past

        xattn_past_beam = tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in xattn_past
        )

        lm_past_beam = tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in lm_past
        )

        return xattn_past_beam, lm_past_beam

