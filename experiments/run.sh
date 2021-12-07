# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------

epoch=2000
save_epoch=400
image_path=../data/etc/man256.png
image_length=256
beta=1.0
gradient_type=convolve
save_name=1

for gradient_type in convolve
do 
python train.py --model siren \
                             --channel-dim 1 \
                              --hidden-layers 5 \
                              --hidden-features 256 \
                              --gradient-type $gradient_type \
                              --beta $beta \
                              --epochs  $epoch \
                              --plot-epoch $save_epoch \
                              --save-dir third \
                              --image-path $image_path \
                              --image-length  $image_length \
                              --lr 0.01 \
                              --lr-end 0.001 \
                              --plot-full \
                              #--relu-pe-freq 10
done
