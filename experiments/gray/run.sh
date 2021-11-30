# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------

epoch=10
save_epoch=2
image_path=../../data/etc/dog224.png
image_length=224
beta=0.0
gradient_type=convolve
save_name=1

for gradient_type in bumjin
do 
python run_plot_experiment.py --model siren \
                              --channel-dim 1 \
                              --hidden-layers 4 \
                              --hidden-features 224 \
                              --gradient-type $gradient_type \
                              --beta $beta \
                              --epochs  $epoch \
                              --plot-epoch $save_epoch \
                              --save-dir $gradient_type \
                              --image-path $image_path \
                              --image-length  $image_length \
                              --lr 0.01 \
                              --lr-end 0.05 \
                              --plot-full

done

