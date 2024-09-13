python -m torch.distributed.launch --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --batch_size 16 \
    --world_size 1 \
    --model vit_large_patch16 \
    --epochs 100 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 2 \
    --data_path ./DDI_HAM10000/ \
    --task ./finetune_HAM10000/ \
    --finetune ./RETFound_oct_weights.pth \
    # --finetune ./checkpoint-best.pth
    --input_size 224
