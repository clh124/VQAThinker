# VQAThinker: Exploring Generalizable and Explainable Video Quality Assessment via Reinforcement Learning



## 🛠️ Setup

```bash
conda create -n vqathinker python=3.11
conda activate vqathinker
bash setup.sh
```

## 💪🏻 Training

First, replace the `modeling_internvl_chat.py` file under the `InternVL3-8B/` checkpoint directory with the one provided in the current directory.


```bash
bash run_scripts/run_grpo_lsvq.sh
```bash


## For your own data

The jsonl has the format as follows:

```json
{
  "id": 0, 
  "dataset_name": "LSVQ", 
  "image": ["yfcc-batch16/13078.mp4"], 
  "conversations": [
    {"from": "human", "value": "You are doing the video quality assessment task. Here is the question: What is your overall rating on the quality of this video? The rating should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality."}, 
    {"from": "gpt", "value": 3.418571}
    ]
}
```
<!-- 
## 🤝 Acknowledgements

We would like to express our sincere gratitude to [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [R1-V](https://github.com/Deep-Agent/R1-V), [RefCOCO](https://github.com/lichengunc/refer), [RefGTA](https://github.com/mikittt/easy-to-understand-REG/tree/master/pyutils/refer2), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [OVDEval](https://github.com/om-ai-lab/OVDEval), [GUI-Testing-Arena](https://huggingface.co/datasets/songjah/GTArena-UI-Defects), and [LISA](https://github.com/dvlab-research/LISA) for providing open-source resources that contributed to the development of this project.

## ⭐️ Citation

If you find this project useful, welcome to cite us.

```bib
@article{shen2025vlm,
  title={Vlm-r1: A stable and generalizable r1-style large vision-language model},
  author={Shen, Haozhan and Liu, Peng and Li, Jingcheng and Fang, Chunxin and Ma, Yibo and Liao, Jiajia and Shen, Qiaoli and Zhang, Zilun and Zhao, Kangjia and Zhang, Qianqian and Xu, Ruochen and Zhao, Tiancheng },
  journal={arXiv preprint arXiv:2504.07615},
  year={2025}
}
``` -->
