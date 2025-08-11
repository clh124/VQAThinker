<div align="center">

# VQAThinker: Exploring Generalizable and Explainable Video Quality Assessment via Reinforcement Learning

 <div>
    <a href="https://arxiv.org/pdf/2508.06051"><img src="https://img.shields.io/badge/Arxiv-2411.03795-blue"/></a>
    <a href="https://huggingface.co/kkkkkklinhan/InternVL3-VQAThinker-8B"><img src="https://img.shields.io/badge/Model-Release-orange"></a>
   </div>


This is the official code of VQAThinker, the first open-source NR-VQA model enhanced via reinforcement learning, capable of performing both video quality scoring and understanding.

<p align="center">
    <img src="images/performance.png" style="max-width:100%; height:auto;">
</p>

  <div>
      <a href="https://scholar.google.com/citations?user=WmE6necAAAAJ&hl=zh-CN" target="_blank">Linhan Cao</a><sup>1</sup>,
      <a href="https://scholar.google.com/citations?hl=zh-CN&user=nDlEBJ8AAAAJ" target="_blank">Wei Sun</a><sup>2</sup><sup>*</sup><sup>#</sup>,
      <a href="https://scholar.google.com/citations?hl=zh-CN&user=KK2nLnQAAAAJ" target="_blank">Weixia Zhang</a><sup>1</sup>,
      <a href="https://scholar.google.com/citations?hl=zh-CN&user=k7YfbnEAAAAJ" target="_blank">Xiangyang Zhu</a><sup>3</sup>,
      Jun Jia<sup>1</sup>,
  </div>

<div>
      Kaiwei Zhang<sup>1</sup>,
      <a href="https://faculty.ecnu.edu.cn/_s47/zdd/list.psp" target="_blank">Dandan Zhu</a><sup>2</sup>,
      <a href="https://ee.sjtu.edu.cn/en/FacultyDetail.aspx?id=24&infoid=153&flag=153" target="_blank">Guangtao Zhai</a><sup>1</sup>
      <a href="https://scholar.google.com/citations?user=91sjuWIAAAAJ&hl=zh-CN&oi=ao" target="_blank">Xiongkuo Min</a><sup>1</sup><sup>#</sup>,
      
  </div>

  <div>
  <sup>1</sup>Shanghai Jiaotong University,  <sup>2</sup>East China Normal University, <sup>3</sup>Shanghai Artificial Intelligence Laboratory
       </div>   
<div>
<sup>*</sup>Project lead. <sup>#</sup>Corresponding authors. 


<p align="center">
    <img src="images/model.png" style="max-width:100%; height:auto;">
</p>


<div align="left">

## Release
- [08/11/25] 🤗 Released the inference code.


## Installation

```bash
conda create -n vqathinker python=3.11
conda activate vqathinker
bash setup.sh
```

## Quick Inference

Single video quality evaluation:
```shell
python single_infer.py
```

Batch videos quality evaluation:
```shell
python batch_infer.py
```
