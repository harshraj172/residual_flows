## Setup
Activate environmet through yml file
```
conda env create --file environment.yml
```

## Evaluate CIFAR-10 + CIFAR-10(Blurred)
```
python residual_flows/train_img.py --data cifar10 --eval_model True --imagesize 32 --actnorm True --wd 0  --resume residual_flows/models/cifar10_resflow_16-16-16.pth --val-batchsize 1 --block resblock --do_hierarch True
```
