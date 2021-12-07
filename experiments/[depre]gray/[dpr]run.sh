# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------

epoch=600
save_epoch=150
image_path=../../data/Set14/man.png
image_length=512
beta=0.0
gradient_type=convolve
save_name=1

for gradient_type in convolve
do 
python train.py --model siren \
                              --channel-dim 1 \
                              --hidden-layers 5 \
                              --hidden-features 512 \
                              --gradient-type $gradient_type \
                              --beta $beta \
                              --epochs  $epoch \
                              --plot-epoch $save_epoch \
                              --save-dir $gradient_type \
                              --image-path $image_path \
                              --image-length  $image_length \
                              --lr 0.01 \
                              --lr-end 0.001 \
                              --plot-full
done

