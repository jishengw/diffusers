exec >logfile.txt 2>&1 # Log the script for debugging
export MODEL_NAME='stabilityai/stable-diffusion-2-1'
export INSTANCE_DIR="./train_image_two_people" # The folder for training image dataset
export CLASS_DIR="./class_two_people" # The folder that store Class image for Prior preservation loss
export OUTPUT_DIR="./model_two_people" # The folder that store the fine tuned model


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="2 people" \
  --class_prompt="people" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=10 \
  --num_class_images=20 \
  --max_train_steps=200 \
  --checkpointing_steps=100 \



#accelerate launch train_dreambooth.py \
#  --pretrained_model_name_or_path=$MODEL_NAME  \
#  --instance_data_dir=$INSTANCE_DIR \
#  --class_data_dir=$CLASS_DIR \
#  --output_dir=$OUTPUT_DIR \
#  --with_prior_preservation --prior_loss_weight=1.0 \
#  --instance_prompt="a photo of sks dog" \
#  --class_prompt="a photo of dog" \
#  --resolution=512 \
#  --train_batch_size=1 \
#  --gradient_accumulation_steps=2 --gradient_checkpointing \
#  --use_8bit_adam \
#  --learning_rate=5e-6 \
#  --lr_scheduler="constant" \
#  --lr_warmup_steps=0 \
#  --num_class_images=200 \
#  --max_train_steps=800
#
#


#accelerate launch train_dreambooth.py \
#  --pretrained_model_name_or_path=$MODEL_NAME  \
#  --instance_data_dir=$INSTANCE_DIR \
#  --output_dir=$OUTPUT_DIR \
#  --instance_prompt="a photo of sks dog" \
#  --resolution=512 \
#  --train_batch_size=1 \
#  --gradient_accumulation_steps=1 \
#  --learning_rate=5e-6 \
#  --lr_scheduler="constant" \
#  --lr_warmup_steps=0 \
#  --max_train_steps=400 \
#  --gradient_checkpointing \
#  --use_8bit_adam \

