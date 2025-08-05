PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"
# on remote
data_paths="/tos-bjml-researcheval/wenfarong/caolinhan/data/LSVQ/labels_train.jsonl" 
image_folders="/tos-bjml-researcheval/wenfarong/caolinhan/data/LSVQ/"
model_path="/dev/shm/InternVL3-8B/"
is_reward_customized_from_vlm_module=False
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"

export EXP_NAME="InternVL3-8B_GRPO" # TODO: change this to your own experiment name
TASK_TYPE="rec"
cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
# MAX_STEPS=1200 # TODO: change this to your own max steps


# export WANDB_DISABLED=true
# CUDA_VISIBLE_DEVICES=4,5,6,7
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir /dev/shm/internvl3_8B_lsvq_grpo \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --max_anyres_num 6 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 3 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 50 \
    --save_total_limit 1 \
    --num_generations 4 \
    --max_completion_length 2048 \
    --reward_funcs accuracy format diversity \
    --beta 0.04 \
    --report_to none \
    --dataset-name this_is_not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero2.json \

echo "Training completed for ${EXP_NAME}"