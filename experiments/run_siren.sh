# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------

epoch=3000
save_epoch=300
image_path=../data/etc/dog224.png
image_length=224
beta=0.0
gradient_type=sobel
save_name=$gradient_type

python run_plot_experiment.py --model siren \
                              --hidden-layers 4 \
                              --hidden-features 224 \
                              --gradient-type $gradient_type \
                              --beta $beta \
                              --epochs  $epoch \
                              --plot-epoch $save_epoch \
                              --save-dir $save_name \
                              --image-path $image_path \
                              --image-length  $image_length \
                              --lr 0.01 \
                              --lr-end 0.001 \
                              --plot-full



