###################### e.g.: coco overfitting training ######################################

export MODEL_NAME="xxxxxxxxxxx"
export dataset_name="xxxxxxxxxxx"

CUDA_VISIBLE_DEVICES=0 nohup accelerate launch  --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512  \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=150000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="coco_m_overfitting" \
  > log.txt &

###################### e.g.: coco real-world training ######################################

export MODEL_NAME="xxxxxxxxxxx"
export dataset_name="xxxxxxxxxxx"

CUDA_VISIBLE_DEVICES=0 nohup accelerate launch  --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=50000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="coco_m_real" \
  > log.txt &