# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------

epoch=1000
save_epoch=200
image_path=../../data/etc/eagle256.jpg
image_path2=../../data/etc/background256.jpg
image_length=256

beta=0.0
alpha=0.5
gradient_type=convolve

for gradient_type in convolve
do 
save_name=$gradient_type
python run_composition_experiment.py --model siren \
                             --channel-dim 1 \
                              --hidden-layers 5 \
                              --hidden-features 256 \
                              --gradient-type $gradient_type \
                              --beta $beta \
                              --alpha $alpha \
                              --epochs  $epoch \
                              --plot-epoch $save_epoch \
                              --save-dir $gradient_type \
                              --image-path $image_path \
                              --image-path2 $image_path2 \
                              --image-length  $image_length \
                              --lr 0.01 \
                              --lr-end 0.001 \
                              --plot-full \
                              #--relu-pe-freq 10

done

