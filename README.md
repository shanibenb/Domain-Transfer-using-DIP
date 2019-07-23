# Domain-Transfer-using-DIP
## Abstract:
The goal of this project, is to map a single image from a source domain A, to a target domain B, which contains a set of images.
For more information check the file ["Project.pdf"](https://github.com/shanibenb/Domain-Transfer-using-DIP/blob/master/Project.pdf).

## How to run?
**run MNIST to SHVH transformation:**
1. To train the first phase run: 
python main_mnist_to_svhn_autoencoder.py
You can skip this phase by using a pre-trained model in the results folder.
2. To create a new image translated from MNIST to SVHN run: 
python main_mnist_to_svhn.py --mode=train
(images will be saved automatically at sample_path, if you wish to plot after every sample_step add --PLOT=True)
3. To test the network first download ["svhn_classifier.pkl"](https://drive.google.com/file/d/1OK5ifdDXaxquOCklUup32B0a3QN3vaJj/view?usp=sharing)
and then run: python main_mnist_to_svhn.py --mode=test

**run MNIST to SHVH transformation:**
1. To train the first phase run: 
python main_svhn_to_mnist_autoencoder.py
You can skip this phase by using a pre-trained model in the results folder.
2. To create a new image translated from SVHN to MNIST run: 
python main_svhn_to_mnist.py --mode=train
(images will be saved automatically at sample_path, if you wish to plot after every sample_step add --PLOT=True)
3. To test the network run:
python main_svhn_to_mnist.py --mode=test

**run Summer to Winter Yosemite transformation:**
1. To train the first phase run:
python train.py --dataroot=./datasets/summer2winter_yosemite/trainB --name=summer2winter_yosemite_autoencoder --model=DIP_AE --dataset_mode=single
You can skip this phase by using a pre-trained model in the checkpoints folder.
2. To create a new image translated from MNIST to SVHN run:
python train.py --dataroot=./datasets/summer2winter_yosemite/ --name=summer2winter_yosemite_DIP --load_dir=summer2winter_yosemite_autoencoder --model=DIP --start=0 --max_items_A=1
3. To test the network run:
python test.py --dataroot=./datasets/summer2winter_yosemite/ --name=summer2winter_yosemite_DIP --load_dir=summer2winter_yosemite_autoencoder --model=DIP --start=0 --max_items_A=1

**run Winter to Summer Yosemite transformation:**
1. To train the first phase run:
python train.py --dataroot=./datasets/summer2winter_yosemite/trainA --name=summer2winter_yosemite_autoencoder_reverse --model=DIP_AE --dataset_mode=single
2. To create a new image translated from MNIST to SVHN run:
python train.py --dataroot=./datasets/summer2winter_yosemite/ --name=summer2winter_yosemite_DIP_reverse --load_dir=summer2winter_yosemite_autoencoder_reverse --model=DIP --start=0 --max_items_A=1 --A=’B’ --B=’A’
3. To test the network run:
python test.py --dataroot=./datasets/summer2winter_yosemite/ --name=summer2winter_yosemite_DIP_reverse --load_dir=summer2winter_yosemite_autoencoder_reverse --model=DIP --start=0 --max_items_A=1 --A=’B’ --B=’A’

**Additional downloads**
To download dataset for Winter-Summer Yosemite: bash datasets/download_cyclegan_dataset.sh $summer2winter_yosemite where DATASET_NAME is one of (facades, cityscapes, maps, monet2photo, summer2winter_yosemite)

To visualize losses: run python -m visdom.server
