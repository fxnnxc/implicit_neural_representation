# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------

epoch=1001
save_epoch=250
bias_epoch=1000
save_bias_epoch=250
image_path=../data/etc/man256.png
image_length=256
beta=0.0
gradient_type=convolve
save_name=1

for gradient_type in convolve
do 
python train.py --model siren \
                --eoren \
                --channel-dim 1 \
                --hidden-layers 5 \
                --hidden-features 256 \
                --gradient-type $gradient_type \
                --beta $beta \
                --epochs  $epoch \
                --bias-epochs  $bias_epoch \
                --plot-bias-epoch  $save_bias_epoch \
                --plot-epoch $save_epoch \
                --save-dir third \
                --image-path $image_path \
                --sidelength  $image_length \
                --lr 0.01 \
                --lr-end 0.001 \
                --plot-full \
                # --relu-pe-freq 10
done
