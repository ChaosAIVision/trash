CUDA_VISIBLE_DEVICES=1 python -m pythera.sdx.arch.inpaint_catvton.trainer \
 --pretrained_model_name_or_path="botp/stable-diffusion-v1-5-inpainting" \
 --output_dir="/home/tiennv/trang/chaos/embedding_data/save_checkpoint_catvton" \
 --resolution=512 \
 --mode 0 \
 --learning_rate=5e-6 \
 --adam_weight_decay 1e-2 \
 --gradient_accumulation_steps 8 \
 --checkpoints_total_limit 5 \
 --checkpointing_steps 500 \
 --num_train_epochs 200 \
 --train_batch_size=1 \
 --mixed_precision 'bf16' \
  --dataset_path '/home/tiennv/trang/chaos/controlnext/data/datatest/data_high_quality.csv' \
  --embedding_dir '/home/tiennv/trang/chaos/embedding_data/embedding_catvton' \
  --input_type 'csv' \
  --resume_from_checkpoint 'latest' \
  --prediction_type epsilon \
  --use_adam8bit True \
    # --save_embeddings_to_npz True \

#  --unet_model_name_or_path '/home/tiennv/trang/chaos/controlnext/weight_pretrain/unet_inpainting' \
