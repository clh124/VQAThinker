# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union

import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from pytorchvideo.models.hub import slowfast_r50


logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


def pack_pathway_output(frames, device):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    fast_pathway = frames  # Shape: [B, C, T, H, W]

    # Generate the index tensor on the same device as 'frames'
    index = torch.linspace(
        0, frames.shape[2] - 1, frames.shape[2] // 3
    ).long().to(frames.device)

    # Perform temporal sampling from the fast pathway
    slow_pathway = frames.index_select(2, index)
    
    # print(slow_pathway.shape)

    frame_list = [slow_pathway.to(device), fast_pathway.to(device)]

    # fast_pathway = frames
    # # Perform temporal sampling from the fast pathway.
    # slow_pathway = torch.index_select(
    #     frames,
    #     2,
    #     torch.linspace(
    #         0, frames.shape[2] - 1, frames.shape[2] // 4
    #     ).long(),
    # )
    # frame_list = [slow_pathway.to(device), fast_pathway.to(device)]

    return frame_list

class slowfast(torch.nn.Module):
    def __init__(self):
        super(slowfast, self).__init__()
        # Initialize the model without pretrained weights
        #slowfast_model = slowfast_r50(pretrained='/root/autodl-tmp/InternVL/internvl_chat/SLOWFAST_8x8_R50.pkl')

        # Load the local checkpoint
        # checkpoint_path = '/root/autodl-tmp/InternVL/internvl_chat/SLOWFAST_8x8_R50.pkl'  # Replace with your actual path
        # checkpoint = torch.load(checkpoint_path,encoding='latin1')

        # # If your checkpoint contains 'state_dict', adjust accordingly
        # if 'state_dict' in checkpoint:
        #     state_dict = checkpoint['state_dict']
        # else:
        #     state_dict = checkpoint

        # # Remove 'module.' prefix if necessary
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if k.startswith('module.'):
        #         new_state_dict[k[7:]] = v
        #     else:
        #         new_state_dict[k] = v

        # # Load the state dictionary into the model
        # slowfast_model.load_state_dict(new_state_dict, strict=False)

        # Now extract the features as before
        #slowfast_pretrained_features = nn.Sequential(*list(slowfast_model.children())[0])
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0,5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)
        

    def forward(self, x):
        with torch.no_grad():
            
            x = self.feature_extraction(x)
            x[0] = x[0].repeat_interleave(6, dim=2)
            x[1] = x[1].repeat_interleave(6, dim=2)
            slow_feature = self.slow_avg_pool(x[0])
            fast_feature = self.fast_avg_pool(x[1])

            slow_feature = self.adp_avg_pool(slow_feature)
            fast_feature = self.adp_avg_pool(fast_feature)
            # print(slow_feature.shape) #[2,2048,1,1,1]
            # print(fast_feature.shape) #[2,256,1,1,1]
            feature_3D = torch.cat([slow_feature, fast_feature],dim=1) #[2,2304,1,1,1]
            # feature_3D = fast_feature #[2,256,1,1,1]
        return feature_3D


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'Qwen2DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = 256
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.motion_mlp = nn.Sequential(
            nn.LayerNorm(2304),
            nn.Linear(2304, 3584),
            nn.GELU(),
            nn.Linear(3584, 3584)
        )
        
        for m in self.motion_mlp.modules():
            if isinstance(m, nn.Linear):
                print('motion_mlp.weight1',m.weight)
                m.weight.data.uniform_(0.0, 1e-2)
                print('motion_mlp.weight2',m.weight)
                m.bias.data.zero_()
                print('motion_mlp.bias',m.bias)

        self.slowfast_model = slowfast()

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        frames = pixel_values.view(B, int(vit_batch_size/B), 3, 448, 448)
        # print("frames1", frames.shape)
        frames = frames.permute(0, 2, 1, 3, 4)
        # print("frames2", frames.shape)
        device = pixel_values.device

        inputs = pack_pathway_output(frames, device)  # Returns [slow_pathway, fast_pathway]
        # print("inputs", inputs[0].shape, inputs[1].shape)
        motion_feature = self.slowfast_model(inputs)
        # print("motion_feature", motion_feature)
        # 判断 motion_feature 是否全为 0（支持 tensor 类型）
        if torch.all(motion_feature == 0):
            print("⚠️ Warning: Detected zero motion feature!")
            print("Input shape:", inputs.shape)
            print("Input tensor:", inputs)

        motion_feature = motion_feature.view(B, -1)
        # print("motion_feature2", motion_feature.shape)
        motion_embeds = self.motion_mlp(motion_feature)
        # print("motion_embeds", motion_embeds.shape)


        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        # input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        dim = 1 if selected.dim() > 1 else 0
        selected_cumsum = torch.cumsum(selected, dim=dim)
        # print("selected_cumsum", selected_cumsum)

        # Find the maximum cumulative sum per batch
        max_cumsum = selected_cumsum.max(dim=dim, keepdim=True)[0]

        # Create a mask for the last True position in each batch
        last_true_mask = (selected_cumsum == max_cumsum) & selected
        # print("last_true_mask", last_true_mask)
    
        # Create 'selected1' by setting the last True in 'selected' to False
        selected1 = selected.clone()
        selected1[last_true_mask] = False

        # Create 'selected2' by keeping only the last True position per batch
        selected2 = last_true_mask
    

        # Reshape back to (B*N)
        selected1 = selected1.view(B * N)
        selected2 = selected2.view(B * N)

        input_ids = input_ids.reshape(B * N)


        try:
            # input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            input_embeds[selected1] = input_embeds[selected1] * 0.0 + vit_embeds.reshape(-1, C)
            input_embeds[selected2] = input_embeds[selected2] * 0.0 + motion_embeds.reshape(-1, C)
            
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            # print("forward forward!!!")
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        # assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * int(self.num_image_token * num_patches) + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_batch_size = pixel_values.shape[0]
                # print("pixel_values", pixel_values.shape)
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            frames = pixel_values.view(B, int(vit_batch_size/B), 3, 448, 448)
            # print("frames1", frames.shape)
            frames = frames.permute(0, 2, 1, 3, 4)
            # print("frames2", frames.shape)
            device = pixel_values.device

            inputs = pack_pathway_output(frames, device)  # Returns [slow_pathway, fast_pathway]
            # print("inputs", inputs[0].shape, inputs[1].shape)
            motion_feature = self.slowfast_model(inputs)
            # print("motion_feature", motion_feature.shape)
            motion_feature = motion_feature.view(B, -1)
            # print("motion_feature2", motion_feature.shape)
            motion_embeds = self.motion_mlp(motion_feature)
            # print("motion_embeds", motion_embeds.shape)




            # input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)

            dim = 1 if selected.dim() > 1 else 0
            selected_cumsum = torch.cumsum(selected, dim=dim)
            # print("selected_cumsum", selected_cumsum)

            # Find the maximum cumulative sum per batch
            max_cumsum = selected_cumsum.max(dim=dim, keepdim=True)[0]

            # Create a mask for the last True position in each batch
            last_true_mask = (selected_cumsum == max_cumsum) & selected
            # print("last_true_mask", last_true_mask)
        
            # Create 'selected1' by setting the last True in 'selected' to False
            selected1 = selected.clone()
            selected1[last_true_mask] = False

            # Create 'selected2' by keeping only the last True position per batch
            selected2 = last_true_mask
        

            # Reshape back to (B*N)
            selected1 = selected1.view(B * N)
            selected2 = selected2.view(B * N)

            input_ids = input_ids.reshape(B * N)
            count = selected.sum().item()
            assert selected.sum() != 0
            # print("vit_embeds", vit_embeds.shape)
            # input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            input_embeds[selected1] = input_embeds[selected1] * 0.0 + vit_embeds.reshape(-1, C)
            input_embeds[selected2] = input_embeds[selected2] * 0.0 + motion_embeds.reshape(-1, C)
            
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # print("input_embeds", input_embeds.shape)
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
