# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------

epoch=2000
save_epoch=500
image_path=../data/etc/dog224.png
image_length=224
beta=0.99

python run_plot_experiment.py --model relu_pe \
                              --hidden-layers 3 \
                              --hidden-features 224 \
                              --beta $beta \
                              --epochs $epoch \
                              --save-epoch $save_epoch \
                              --save-dir test1 \
                              --image-path $image_path \
                              --image-length  $image_length \
                              --lr 0.01 \
                              --lr-end 0.01 \
                              --relu-pe-freq 10



