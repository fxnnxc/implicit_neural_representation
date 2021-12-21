# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------
epoch=1950
save_epoch=950
bias_epoch=50
save_bias_epoch=50
beta=0.0
gradient_type=convolve
save_name=Set5_eoren_pixel

for image_path in ./../data/Set5/*
do 
python train.py --model siren \
                --eoren \
                --channel-dim 1 \
                --hidden-layers 5 \
                --hidden-features 256 \
                --sidelength 256 \
                --gradient-type $gradient_type \
                --beta $beta \
                --epochs  $epoch \
                --bias-epochs  $bias_epoch \
                --plot-bias-epoch  $save_bias_epoch \
                --plot-epoch $save_epoch \
                --save-dir $save_name \
                --image-path $image_path \
                --lr 0.01 \
                --lr-end 0.005 \
                --plot-full \
                # --relu-pe-freq 10
done