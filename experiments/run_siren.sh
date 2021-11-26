# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------

epoch=100
save_epoch=20
image_path=../data/etc/dog224.png
image_length=224
beta=0.98
save_name=test-each

python run_plot_experiment.py --model siren \
                              --hidden-layers 3 \
                              --hidden-features 224 \
                              --beta $beta \
                              --epochs  $epoch \
                              --save-epoch $save_epoch \
                              --save-dir $save_name \
                              --image-path $image_path \
                              --image-length  $image_length \
                              --lr 0.01 \
                              --lr-end 0.01 \
                              --plot-each



