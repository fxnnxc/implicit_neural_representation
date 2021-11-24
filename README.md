# neural_implicit_representation

```
pip install -e .
```


## Requirements 

```
pip install opencv-python
```


## Run experiment

```
cd experiments 
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
                              --beta 0.1 \
                              --epochs 1000 \
                              --save-epoch 100 \
                              --save-dir test1 \
                              --image-path ../data/dog224.png \
                              --image-length  224 \
                              --lr 0.01 \
                              --lr-end 0.001
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
