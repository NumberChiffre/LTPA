# Learn To Pay Attention (ICLR 2018) Using Pytorch

~~~~
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements
python train.py --attention_mode dp --epochs 300 --batch_size 256 --lr 0.1 --logs_path VGG19_CIFAR10
