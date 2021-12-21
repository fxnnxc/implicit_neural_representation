# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------
epoch=950
save_epoch=950
bias_epoch=50
save_bias_epoch=50
beta=0.0
gradient_type=convolve
save_name=DIV_eoren_pixel


for i in 0 1 2 3 4 5 6 7 8 9
do
	save_name=MNIST_${i}_eoren_pixel
	count=1
	for image_path in ./../data/mnist_png/mnist_png/testing/${i}/*
	do 
        python train.py --model siren \
                        --eoren \
                        --channel-dim 1 \
                        --hidden-layers 5 \
                        --hidden-features 256 \
                        --sidelength 256 \
                        --gradient-type $gradient_type \
                        --beta $beta \
                        --epochs  $epoch \
                        --bias-epochs  $bias_epoch \
                        --plot-bias-epoch  $save_bias_epoch \
                        --plot-epoch $save_epoch \
                        --save-dir $save_name \
                        --image-path $image_path \
                        --lr 0.01 \
                        --lr-end 0.01 \
                        --plot-full \
                    # --relu-pe-freq 10
        count=$((count +1))
        echo $count
        if [ $count -eq 10 ]
            then 
            break
        fi
	done
	
done
