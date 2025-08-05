from open_r1.vlm_modules.vlm_module import VLMBaseModule
from typing import Dict, Any, Union
from transformers import AutoModel, AutoProcessor, AutoConfig
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers.feature_extraction_sequence_utils import BatchFeature
from decord import VideoReader, cpu
import numpy as np


IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

import torch
import random

# def apply_temporal_perturbation(pixel_values, mode="shuffle", window_size=4):
#     """
#     对 pixel_values (shape: [T, C, H, W]) 进行时间扰动
#     """
#     T = pixel_values.size(0)

#     if mode == "shuffle":
#         indices = torch.randperm(T)
#         return pixel_values[indices]

#     elif mode == "local_shuffle":
#         new_frames = []
#         for i in range(0, T, window_size):
#             window = pixel_values[i:i+window_size]
#             shuffled = window[torch.randperm(len(window))]
#             new_frames.append(shuffled)
#         return torch.cat(new_frames, dim=0)

#     elif mode == "reverse":
#         return pixel_values.flip(0)

#     elif mode == "jitter":
#         jittered = []
#         for i in range(T):
#             offset = random.choice([-1, 0, 1])
#             j = min(max(i + offset, 0), T - 1)
#             jittered.append(pixel_values[j].unsqueeze(0))
#         return torch.cat(jittered, dim=0)

#     elif mode == "duplicate":
#         T = pixel_values.size(0)
        
#         # 1. 随机选择一个帧的位置作为静止帧
#         duplicate_index = random.randint(0, T - 1)
#         frame_to_duplicate = pixel_values[duplicate_index].unsqueeze(0)
        
#         # 2. 随机决定重复次数（比如重复1~3次）
#         n_repeat = random.randint(1, min(3, T - 1))  # 防止删帧时不足
        
#         repeated_frames = frame_to_duplicate.repeat(n_repeat, 1, 1, 1)

#         # 3. 随机插入位置
#         insert_pos = random.randint(0, T)

#         # 4. 插入重复帧
#         new_pixel_values = torch.cat([
#             pixel_values[:insert_pos],
#             repeated_frames,
#             pixel_values[insert_pos:]
#         ], dim=0)  # shape: [T + n_repeat, C, H, W]

#         # 5. 为保持总帧数不变，从新序列中随机删去 n_repeat 个不同帧（不删重复帧）
#         total_frames = new_pixel_values.size(0)
#         all_indices = list(range(total_frames))

#         # 避免删刚插入的帧（也可以删，只要不全部删）
#         duplicate_range = list(range(insert_pos, insert_pos + n_repeat))
#         candidate_indices = [i for i in all_indices if i not in duplicate_range]

#         if len(candidate_indices) >= n_repeat:
#             drop_indices = random.sample(candidate_indices, n_repeat)
#         else:
#             # 不够删就删一部分插入帧
#             drop_indices = random.sample(all_indices, n_repeat)

#         mask = torch.ones(total_frames, dtype=torch.bool)
#         for idx in drop_indices:
#             mask[idx] = False

#         final_pixel_values = new_pixel_values[mask]
#         return final_pixel_values

#     else:
#         return pixel_values  # 默认不扰动


def apply_temporal_perturbation(pixel_values, mode="shuffle", window_size=4):
    """
    对 pixel_values (shape: [T, C, H, W]) 进行时间扰动
    """
    T = pixel_values.size(0)

    if mode == "shuffle":
        indices = torch.randperm(T)
        return pixel_values[indices]

    elif mode == "local_shuffle":
        new_frames = []
        for i in range(0, T, window_size):
            window = pixel_values[i:i+window_size]
            shuffled = window[torch.randperm(len(window))]
            new_frames.append(shuffled)
        return torch.cat(new_frames, dim=0)

    elif mode == "reverse":
        return pixel_values.flip(0)

    elif mode == "jitter":
        jittered = []
        for i in range(T):
            offset = random.choice([-1, 0, 1])
            j = min(max(i + offset, 0), T - 1)
            jittered.append(pixel_values[j].unsqueeze(0))
        return torch.cat(jittered, dim=0)

    elif mode == "duplicate":
        duplicate_index = random.randint(0, T - 1)
        frame_to_duplicate = pixel_values[duplicate_index].unsqueeze(0)
        n_repeat = random.randint(1, min(3, T - 1))
        repeated_frames = frame_to_duplicate.repeat(n_repeat, 1, 1, 1)
        insert_pos = random.randint(0, T)
        new_pixel_values = torch.cat([
            pixel_values[:insert_pos],
            repeated_frames,
            pixel_values[insert_pos:]
        ], dim=0)
        total_frames = new_pixel_values.size(0)
        all_indices = list(range(total_frames))
        duplicate_range = list(range(insert_pos, insert_pos + n_repeat))
        candidate_indices = [i for i in all_indices if i not in duplicate_range]
        if len(candidate_indices) >= n_repeat:
            drop_indices = random.sample(candidate_indices, n_repeat)
        else:
            drop_indices = random.sample(all_indices, n_repeat)
        mask = torch.ones(total_frames, dtype=torch.bool)
        for idx in drop_indices:
            mask[idx] = False
        final_pixel_values = new_pixel_values[mask]
        return final_pixel_values

    elif mode == "random_drop":
        n_drop = random.randint(1, 2)  # 随机选择 1 到 T/4 帧丢弃
        drop_indices = random.sample(range(T), n_drop)
        dropped_pixel_values = pixel_values.clone()
        for idx in drop_indices:
            dropped_pixel_values[idx] = torch.zeros_like(dropped_pixel_values[idx])  # 置为全黑
        return dropped_pixel_values

    else:
        return pixel_values  # 默认不扰动

class InvernVLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()
        self.conv_template = None
        self.num_image_token = None

    def get_vlm_key(self):
        return "internvl"
        
    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        assert "InternVL" in model_id, f"model_id must contain 'InternVL', but got {model_id}"
        self.model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # The model class of InternVL when being mapped has been determined by its config
        model_cls = AutoModel
        # InternVL should be inputted with "trust_remote_code=True"
        model_init_kwargs["trust_remote_code"] = True
        # "use_cache" should be removed
        model_init_kwargs.pop("use_cache", None)
        # "flash_attention_2" should be modified to "use_flash_attn" in InternVL
        if "flash_attention_2" in model_init_kwargs.get("attn_implementation", ""):
            model_init_kwargs["use_flash_attn"] = True
            model_init_kwargs.pop("attn_implementation")
        return model_cls

    def post_model_init(self, model, processing_class):
        # print("model is", model)
        # print("model type is", type(model))
        self.conv_template = model.conv_template if self.conv_template is None else self.conv_template
        self.num_image_token = model.num_image_token if self.num_image_token is None else self.num_image_token
        img_context_token_id = processing_class.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id
    
    def is_embeds_input(self):
        return True

    def get_processing_class(self):
        return AutoProcessor
    
    def get_eos_token_id(self, processing_class):
        eos_token_id = processing_class.convert_tokens_to_ids(self.conv_template.sep.strip())
        return eos_token_id
        
    def get_vision_modules_keywords(self):
        return ['vision_model']

    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_flags']
    
    def get_non_generate_params(self):
        return ['image_flags']

    def get_custom_processing_keywords(self):
        return [('None', 'max_anyres_num')]

    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = []
        for example in inputs:
            template = self.conv_template.copy()
            conversation_list = example["prompt"]
            system_message = extract_system_message(conversation_list)
            if system_message is not None:
                template.system_message = system_message
            
            processed_list = process_conversation_list(conversation_list, system_message)
            for i, processed_item in enumerate(processed_list):
                if i % 2 == 0:
                    template.append_message(template.roles[0], processed_item)
                else:
                    template.append_message(template.roles[1], processed_item)
            if len(processed_list) % 2 == 1:
                template.append_message(template.roles[1], None)
            query = template.get_prompt()
            prompts_text.append(query)
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False, is_shuffle=False):
        # Process images
        full_pixel_values = []
        num_patches_list = []
        for img in images:
            # print("img", img)
            # if img.endswith(".mp4"):
            if True:
                pixel_values, num_patch_list = self._load_video(img)
                # if is_shuffle:
                #     indices = torch.randperm(pixel_values.size(0))
                #     # print("indices", indices)
                #     pixel_values = pixel_values[indices]
                
                # ===== 使用方式 =====
                if is_shuffle:
                    mode = random.choice(["shuffle", "local_shuffle", "reverse", "jitter", "duplicate", "random_drop"])
                    pixel_values = apply_temporal_perturbation(pixel_values, mode=mode)
                    # print(f"应用时间扰乱模式: {mode}")
                    # print("pixel_values", pixel_values.shape)

                # print("num_patches_list", num_patch_list)
                # print("pixel_values", pixel_values.shape)
                full_pixel_values.append(pixel_values) 
                num_patches_list.extend(num_patch_list)
                num_patches_list.extend([1])
                # print("num_patches_list", num_patches_list)

            else:
                pixel_values = self._load_image(img, input_size=self.model_config.vision_config.image_size, max_num=processing_class.max_anyres_num)
                # print("pixel_values", pixel_values.shape)
                full_pixel_values.append(pixel_values)
                num_patches_list.append(pixel_values.shape[0])
        full_pixel_values = torch.cat(full_pixel_values, dim=0)
        
        # Process prompts
        queries = []
        image_idx = 0
        # print("prompts_text", prompts_text)
        # print("num_patches_list", num_patches_list)
        for query in prompts_text:
            # print("query", query)
            while "<image>" in query:
                if (image_idx+1) % 7 == 0 and image_idx != 0:
                    num_patches = num_patches_list[image_idx]
                    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patches + IMG_END_TOKEN
                    query = query.replace("<image>", image_tokens, 1)
                else:
                    num_patches = num_patches_list[image_idx]
                    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                    query = query.replace("<image>", image_tokens, 1)
                # print("image_tokens", self.num_image_token)
                total_context_token_count = sum(q.count(IMG_CONTEXT_TOKEN) for q in [query])
                # print("image_idx", image_idx)
                # print("总共的 IMG_CONTEXT_TOKEN 个数为:", total_context_token_count)
                image_idx += 1
            # print("image_idx", image_idx)

            
            queries.append(query)
        assert image_idx == len(num_patches_list)
        
        model_inputs = processing_class(
            queries,
            return_tensors=return_tensors,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
        )
        model_inputs["pixel_values"] = full_pixel_values
        # Only support pure-image data currently (each sample should contain the image)
        model_inputs['image_flags'] = torch.ones(full_pixel_values.shape[0], dtype=torch.long)
        
        model_inputs = BatchFeature(data=model_inputs)

        return model_inputs

    def _load_image(self, image_file, input_size: int=448, max_num:int=12):
        # print("image_file", image_file)
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def _get_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices


    def _load_video(self, video_path, bound=None, input_size=448, max_num=1, num_segments=6):
        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)

        if video_path.lower().endswith('.jpg'):
            img = Image.open(video_path).convert('RGB')
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            # print("img", len(img), img[0])
            for i in range(num_segments):
                num_patches_list.append(pixel_values.shape[0])
                pixel_values_list.append(pixel_values)
        else:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            max_frame = len(vr) - 1
            fps = float(vr.get_avg_fps())
            frame_indices = self._get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
            for frame_index in frame_indices:
                img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
                img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
                pixel_values = [transform(tile) for tile in img]
                pixel_values = torch.stack(pixel_values)
                # print("pixel_values.shape[0]", pixel_values.shape[0])
                num_patches_list.append(pixel_values.shape[0])
                pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list






    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
                # return "{Question} only output the final answer in <answer> </answer> tags."
    
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the InternVL model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\[\d+,\s*\d+,\s*\d+,\s*\d+\].*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
        
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from InternVL model and ground truth bounding box."""
        """Adopt soft iou reward here"""
        import re
        import os
        import json
        from datetime import datetime
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        for content, sol in zip(contents, solution):
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards
    
    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "rec":
                    return InvernVLModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return InvernVLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")


def process_conversation_list(conversation_list, system_message=None, image_newline=True):
    if system_message is not None:
        conversation_list = conversation_list[1:]
    processed_list = []
    
    for item in conversation_list:
        role = item["role"]
        content = item["content"]
        
        if isinstance(content, list):
            overall_str = ""
            for content_item in content:
                if content_item.get("type") == "image":
                    overall_str += "<image>" if not image_newline else "<image>\n"
                elif content_item.get("type") == "text":
                    overall_str += content_item.get("text")
                else:
                    raise ValueError(f"Unsupported content type: {type(content_item)}")
            processed_list.append(overall_str)
        elif isinstance(content, str):
            processed_list.append(content)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
    
    return processed_list

def extract_system_message(conversation_list):
    if conversation_list[0]["role"] == "system":
        if isinstance(conversation_list[0]["content"], list):
            return conversation_list[0]["content"][0]["text"]
        else:
            return conversation_list[0]["content"]
    return None


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images