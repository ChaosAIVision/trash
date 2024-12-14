CUDA_VISIBLE_DEVICES=0 python pythera/sdx/trainer.py \
 --pretrained_model_name_or_path="botp/stable-diffusion-v1-5-inpainting" \
 --output_dir="/home/tiennv/chaos/data/weight_training" \
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
 --unet_model_name_or_path '/home/tiennv/chaos/data/weight_pretrain' \
  --dataset_path '/home/tiennv/trang/chaos/controlnext/data/datatest/dataset_deepfinetune_v2_data.csv' \
  --embedding_dir '/home/tiennv/chaos/save_embedidng' \
  --input_type 'raw' \
  --resume_from_checkpoint 'latest' \
  --save_embeddings_to_npz True \