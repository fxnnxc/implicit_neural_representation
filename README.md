# neural_implicit_representation

```
pip install -e .
```

## 1. Run experiment

```
cd experiments 
sh run_siren.sh
```

## 2. How to use arguments 

```
--image-path  ../path/to/image.png     # the path of the image
--image-length 256                     # the length of the image
--save-dir name  # additional directory name decorator

--plot-epoch   # result print and plot save frequency 
--plot-full    # store the results at the end of the training in a single image file 
--plot-each    # store the results while training separately


--high-resolution 2.0    # float value to be achieved  e.g.  2.0 results in the X2 image

--beta 0.99        # 0~1 float. 0: gradient  1:pixel 
--epochs 1000      # training epochs 
--lr 0.01          # initial learning rate
--lr-end  0.001    # learning rate scheduling linearly 
--model <NAME>     # (siren, relu_pe, relu)
--relu-pe-freq     # relu position frequency. must be given for the relu_pe model

```

### ReLU_PE

```
python run_plot_experiment.py --model relu_pe \
                              --hidden-layers 3 \
                              --hidden-features 224 \
                              --beta 0.1 \
                              --epochs 1000 \
                              --save-epoch 100 \
                              --save-dir test1 \
                              --image-path ../data/dog224.png \
                              --image-length  224 \
                              --lr 0.01 \
                              --lr-end 0.001 \
                              --relu-pe-freq 5
```

### SIREN

```
python run_plot_experiment.py --model siren \
                              --hidden-layers 3 \
                              --hidden-features 224 \
                              --beta 0.9999 \
                              --epochs 1500 \
                              --save-epoch 300 \
                              --save-dir magic8 \
                              --image-path ../data/dog224.png \
                              --image-length  224 \
                              --lr 0.01 \
                              --lr-end 0.01
```


### ReLU

```
python run_plot_experiment.py --model relu \
                              --hidden-layers 3 \
                              --hidden-features 224 \
                              --beta 0.1 \
                              --epochs 1000 \
                              --save-epoch 100 \
                              --save-dir test1 \
                              --image-path ../data/dog224.png \
                              --image-length  224 \
                              --lr 0.01 \
                              --lr-end 0.001
```



## Requirements 

```
pip install opencv-python
```