CUDA_VISIBLE_DEVICES=1 python pythera/sdx/trainer.py \
 --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
 --output_dir="/weight_training" \
 --resolution=512 \
 --mode 0 \
 --learning_rate=1e-5 \
 --adam_weight_decay 1e-2 \
 --gradient_accumulation_steps 8 \
 --checkpoints_total_limit 5 \
 --checkpointing_steps 500 \
 --num_train_epochs 1000 \
 --train_batch_size=1 \
 --mixed_precision 'bf16' \
  --dataset_path '_data.csv' \
  --embedding_dir 'embedding_folder' \
  --input_type 'raw' \
  --resume_from_checkpoint 'latest' \
  --save_embeddings_to_npz True \
  --prediction_type epsilon \
#  --unet_model_name_or_path '/home/tiennv/trang/chaos/controlnext/weight_pretrain/unet_inpainting' \
