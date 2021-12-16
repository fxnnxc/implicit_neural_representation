# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------
epoch=1000
save_epoch=500
beta=0.0
gradient_type=convolve
save_name=Set14

for image_path in ./../data/Set14/*
do 
python train.py --model siren \
                             --channel-dim 3 \
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
                              --lr-end 0.001 \
                              --plot-full \
                              #--relu-pe-freq 10
done