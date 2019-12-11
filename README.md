# Learn To Pay Attention (ICLR 2018) Using Pytorch

### Run the repo
~~~~
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements
python train.py --attention_mode dp --epochs 300 --batch_size 128 --lr 0.1 --logs_path VGG19_CIFAR10
~~~~

### Run on Google colab
~~~~
from google.colab import drive
drive.mount('/content/drive')
Run the notebook to train
~~~~

