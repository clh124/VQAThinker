import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import csv
import re
import json
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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


MODEL_PATH  = ""
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()



tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
generation_config = dict(max_new_tokens=1024, do_sample=False)


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
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

def load_video(video_path, bound=None, input_size=560, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def normalize_number(num_str):
    try:
        num_str = num_str.replace(',', '')
        return float(num_str)
    except Exception as e:
        print(f"Error converting '{num_str}' to float: {e}")
        return None

video_paths = [
            "/tos-bjml-researcheval/wenfarong/caolinhan/data/test_data/KoNViD_1k/KoNViD_1k_videos/",
            # "/tos-bjml-researcheval/wenfarong/caolinhan/data/LSVQ/",
            # "/tos-bjml-researcheval/wenfarong/caolinhan/data/LSVQ/",
            # "/tos-bjml-researcheval/wenfarong/caolinhan/data/test_data/LIVE_VQC/Video/",
            # "/tos-bjml-researcheval/wenfarong/caolinhan/data/youtube_ugc/",
            # "/tos-bjml-researcheval/wenfarong/caolinhan/data/test_data/live_yt_gaming/",
            # "/tos-bjml-researcheval/wenfarong/caolinhan/data/test_data/cgvds/",
            # "/tos-bjml-researcheval/wenfarong/caolinhan/data/test_data/waterloo_ivc_4k/",
            # "/tos-bjml-researcheval/wenfarong/caolinhan/data/test_data/live_yt_hfr/",
            # "/tos-bjml-researcheval/wenfarong/caolinhan/data/test_data/VDPVE/"
        ]
    


json_prefix = 'json_files/'
jsons = [
            json_prefix + "Konvid-1k_total_ds_score.json",
            # json_prefix + "LSVQ_whole_test_ds_score.json",
            # json_prefix + "LSVQ_whole_test_1080p_ds_score.json",
            # json_prefix + "LIVE-VQC_total_ds_score.json",
            # json_prefix + "youtube_ugc_total.json",
            # json_prefix + "LIVE-YT-Gaming_total_score.json",
            # json_prefix+ "CGVDS_total_score.json",
            # json_prefix + "Waterloo_IVC_4K_total_score.json",
            # json_prefix + "live_hfr_total_score.json",
            # json_prefix + "VDPVE_train_score.json",
        ]

# 存放结果的 CSV 文件的文件夹
csv_output_folder = ""  # 修改为你希望存放 CSV 文件的文件夹

os.makedirs(csv_output_folder, exist_ok=True)

spearmanr1 = []
personr1 = []
iqadata1=[]


batch_size = 16
for image_path, json_ in zip(video_paths, jsons):
    with open(json_) as f:
        iqadata = json.load(f)
        pred_scores, gt_scores = [], []

        csv_filename = os.path.basename(json_).replace(".json", ".csv")  
        csv_output_path = os.path.join(csv_output_folder, csv_filename) 
    
        with open(csv_output_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["filename", "pred_score", "gt_score"]) 


            for start_idx in tqdm(range(0, len(iqadata["annotations"]), batch_size), desc="Batch Inference"):
                batch_data = iqadata["annotations"][start_idx: start_idx + batch_size]

                pixel_values_list = []
                questions = []
                num_patches_lists = []
                gt_batch = []
                video_path_batch = []

                for llddata in batch_data:
                    video_path = image_path + llddata["image_id"]
                    if not os.path.exists(video_path):
                        print(f"File not found: {video_path}")
                        continue

                    pixel_values, num_patches_list = load_video(video_path, num_segments=12, max_num=1)
                    pixel_values = pixel_values.to(torch.bfloat16).cuda()
                    video_prefix = ''.join([f'Frame{i+1}: <image>' for i in range(len(num_patches_list))])
                    num_patches_list.extend([1 / 400])  #  motion 特征占位
                    question = (
                        'Now you will receive a video:' + video_prefix + 'Motion Feature: <image>' +
                        'You are doing the video quality assessment task. Here is the question: '
                        'What is your overall rating on the quality of this video? The rating should be a float between 1 and 5, '
                        'rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality.'
                        'First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.'
                    )
                    pixel_values_list.append(pixel_values)
                    questions.append(question)
                    num_patches_lists.append(num_patches_list)
                    gt_batch.append(llddata["score"])
                    video_path_batch.append(video_path)

                responses = model.batch_chat(
                    tokenizer=tokenizer,
                    pixel_values=torch.cat(pixel_values_list, dim=0),
                    questions=questions,
                    generation_config=generation_config,
                    num_patches_list=num_patches_lists,
                    return_history=False
                )

                for response, gt_score, filename in zip(responses, gt_batch, video_path_batch):
                    print(f'Assistant: {response}')
                    output_ans = extract_answer(response)
                    pred_score = normalize_number(output_ans)
                    if pred_score is None:
                        pred_score = 3.00
                    print(pred_score, gt_score)
                    csv_writer.writerow([filename, pred_score, gt_score])

                    pred_scores.append(float(pred_score))
                    gt_scores.append(float(gt_score))

                if len(pred_scores) > 1:
                    print("Spearmanr", spearmanr(pred_scores, gt_scores)[0],
                        "Pearson", pearsonr(pred_scores, gt_scores)[0])




