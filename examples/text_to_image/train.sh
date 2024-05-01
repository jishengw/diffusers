exec >logfile.txt 2>&1 # Log the script for debugging
export MODEL_NAME='stabilityai/stable-diffusion-2-1'
export OUTPUT_DIR="./lora_two_people"
#export HUB_MODEL_ID="pokemon-lora"
export DATASET_NAME="../dreambooth/flickr30k"

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
  --validation_prompt="two people" \
  --seed=1337