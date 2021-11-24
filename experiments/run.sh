# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------

epoch=50
save_epoch=50
image_path=../data/dog224.png
image_length=224

for beta in 0 0.2 0.8 1.0
do 
# run relu_pe
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
                              --lr-end 0.001 \
                              --relu-pe-freq 5

# run siren
python run_plot_experiment.py --model siren \
                              --hidden-layers 3 \
                              --hidden-features 224 \
                              --beta $beta \
                              --epochs  $epoch \
                              --save-epoch $save_epoch \
                              --save-dir test1 \
                              --image-path $image_path \
                              --image-length  $image_length \
                              --lr 0.01 \
                              --lr-end 0.001
done 
