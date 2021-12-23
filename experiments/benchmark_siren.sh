# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------
epoch=1000
save_epoch=500
beta=0.0
gradient_type=convolve
save_name=Set5_siren_grad

for image_path in ./../data/Set5/*
do
    python train.py --model siren \
                            --channel-dim 1 \
                            --hidden-layers 5 \
                            --hidden-features 256 \
                            --sidelength 256 \
                            --gradient-type $gradient_type \
                            --beta $beta \
                            --epochs  $epoch \
                            --plot-epoch $save_epoch \
                            --save-dir $save_name \
                            --image-path $image_path \
                            --lr 0.01 \
                            --lr-end 0.01 \
                            --plot-full \
                            #--relu-pe-freq 10]	
done
