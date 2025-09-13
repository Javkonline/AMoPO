


NNODES=1
GPUS_PER_NODE=2
WORLD_SIZE=$(($NNODES*$GPUS_PER_NODE))
echo NNODES:${NNODES}
echo GPUS_PER_NODE:${GPUS_PER_NODE}
echo WORLD_SIZE:${WORLD_SIZE}
echo GRAD_ACC_STEPS:$GRAD_ACC_STEPS

GRADIENT_ACCUMULATION_STEPS=1
PRE_DEVICE_TRAIN_BATCH_SIZE=1
TRAIN_BATCH_SIZE=$(($NNODES*$GPUS_PER_NODE*$GRADIENT_ACCUMULATION_STEPS*$PRE_DEVICE_TRAIN_BATCH_SIZE))
FLAG="uniform_sample"
SIMPO_GAMMA=2.0
PREF_BETA=0.8
LEARNING_RATE=3e-7
NUM_EPOCH=3.0

# qwen2.5-32B-ins
Model_PATH="Qwen2.5-0.5B-Instruct"
OUTPUT_DIR=/OUTPUT/AMOPO
echo OUTPUT_DIR:$OUTPUT_DIR

accelerate launch \
--config_file run_amopo/deepspeed_config.yaml \
--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
--num_machines $NNODES \
--num_processes $WORLD_SIZE \
src/train.py \
--stage amopo \
--model_name_or_path $Model_PATH \
--do_train \
--dataset help_steer2_train_format_po2 \
--template qwen \
--finetuning_type full \
--lora_target  all \
--pref_beta  $PREF_BETA \
--simpo_gamma $SIMPO_GAMMA \
--cutoff_len 1024 \
--max_samples  10000 \
--preprocessing_num_workers  16 \
--output_dir  $OUTPUT_DIR \
--overwrite_cache \
--logging_steps  1 \
--save_steps 10000 \
--plot_loss  true \
--overwrite_output_dir  true \
--per_device_train_batch_size  $PRE_DEVICE_TRAIN_BATCH_SIZE \
--gradient_accumulation_steps  $GRADIENT_ACCUMULATION_STEPS \
--learning_rate  $LEARNING_RATE \
--num_train_epochs  $NUM_EPOCH \
--lr_scheduler_type  cosine \
--max_grad_norm 3.0 \
--warmup_ratio  0.1 \
--ddp_timeout  180000 \
--val_size  0.1 \
--per_device_eval_batch_size  1 \
--eval_strategy  steps \
--eval_steps  5000 \
--report_to tensorboard
