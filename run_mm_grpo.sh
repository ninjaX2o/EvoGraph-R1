#!/usr/bin/env bash
set -euo pipefail

path="${POLICY_MODEL_PATH:-}"
model="${POLICY_MODEL_NAME:-}"
dataset="${MM_DATASET:-${DATASET:-}}"
subset="${MM_SUBSET:-${SUBSET:-}}"
mm_data_root="${MM_DATA_ROOT:-.}"

while getopts "p:m:d:s:" opt; do
  case $opt in
    p) path=$OPTARG ;;
    m) model=$OPTARG ;;
    d) dataset=$OPTARG ;;
    s) subset=$OPTARG ;;
    *) echo "Invalid option"; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

if [ -z "$path" ] || [ -z "$model" ] || [ -z "$dataset" ] || [ -z "$subset" ]; then
  echo "Usage: bash run_mm_grpo.sh -p <policy_model_path> -m <model_name> -d <mm_dataset> -s <mm_subset> [hydra overrides...]"
  exit 1
fi

safe_subset="$(printf '%s' "$subset" | tr -c 'A-Za-z0-9_.-' '_')"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export BASE_MODEL="$path"
export PROJECT_NAME="${PROJECT_NAME:-EvoGraph-R1-MM}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${model}_${dataset}_${safe_subset}_mm_grpo}"
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TEXT_SEARCH_API_URL="${TEXT_SEARCH_API_URL:-http://127.0.0.1:8001/search}"
export MM_SEARCH_API_URL="${MM_SEARCH_API_URL:-http://127.0.0.1:8003/search}"
export MM_API_URL="$MM_SEARCH_API_URL"
export WEBSEARCH_CACHE_DATASET="${WEBSEARCH_CACHE_DATASET:-${dataset}_${safe_subset}}"

TRAIN_FILE="${TRAIN_FILE:-${mm_data_root}/datasets_mm/${dataset}/processed/${subset}/train.parquet}"
VAL_FILE="${VAL_FILE:-${mm_data_root}/datasets_mm/${dataset}/processed/${subset}/test.parquet}"

test -f "$TRAIN_FILE"
test -f "$VAL_FILE"

set -x

ray stop
ray start --head

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.val_batch_size="${VAL_BATCH_SIZE:-4}" \
    data.image_key="${IMAGE_KEY:-image_path}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-4}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-4096}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-4096}" \
    data.max_start_length="${MAX_START_LENGTH:-4096}" \
    data.max_tool_response_length="${MAX_TOOL_RESPONSE_LENGTH:-4096}" \
    data.use_custom_tool_format_func=true \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    +actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-5e-7}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-4}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${ACTOR_KL_LOSS_COEF:-0.001}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name="${ROLLOUT_NAME:-vllm}" \
    +actor_rollout_ref.rollout.micro_batch_size="${ROLLOUT_MICRO_BATCH_SIZE:-1}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-2}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN:-null}" \
    actor_rollout_ref.rollout.dtype="${ROLLOUT_DTYPE:-bfloat16}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-1}" \
    actor_rollout_ref.rollout.n_repeat="${ROLLOUT_N_REPEAT:-1}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    "trainer.logger=${TRAINER_LOGGER:-['console']}" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="${N_GPUS:-4}" \
    trainer.nnodes="${NNODES:-1}" \
    trainer.resume_mode="${RESUME_MODE:-auto}" \
    trainer.save_freq="${SAVE_FREQ:--1}" \
    trainer.test_freq="${TEST_FREQ:--1}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-false}" \
    tool.env=mm_all \
    tool.max_turns="${TOOL_MAX_TURNS:-2}" \
    tool.use_batch_tool_calls=True \
    +data.num_workers=0 \
    +data.pin_memory=False \
    +data.persistent_workers=False \
    "$@"
