# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------
epoch=100
save_epoch=10
image_path=../data/Set14/man.png
beta=1.0
gradient_type=convolve
save_name=manman

for gradient_type in convolve
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
                              --save-dir third \
                              --image-path $image_path \
                              --lr 0.01 \
                              --lr-end 0.001 \
                              --plot-full \
                              #--relu-pe-freq 10
done
