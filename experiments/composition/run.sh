# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------
epoch=1500
save_epoch=500
bias_epoch=50
save_bias_epoch=50
image_path=../../data/etc/eagle256.jpg
image_path2=../../data/etc/background256.jpg
image_length=256

beta=0.0
alpha=0.2
gradient_type=convolve

for gradient_type in convolve
do 
save_name=0.2
python run_composition_experiment.py --model siren \
                             --channel-dim 1 \
                              --hidden-layers 5 \
                              --hidden-features 256 \
                              --gradient-type $gradient_type \
                              --beta $beta \
                              --alpha $alpha \
                              --epochs  $epoch \
                              --bias-epochs  $bias_epoch \
                              --plot-bias-epoch  $save_bias_epoch \
                              --plot-epoch $save_epoch \
                              --sidelength 256 \
                              --save-dir $save_name \
                              --image-path $image_path \
                              --image-path2 $image_path2 \
                              --lr 0.01 \
                              --lr-end 0.01 \
                              --plot-full \
                              #--relu-pe-freq 10

done

