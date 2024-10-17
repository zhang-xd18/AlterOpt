import argparse

parser = argparse.ArgumentParser(description='CRNet PyTorch Training')

# ========================== Working mode arguments ==========================
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', type=str, default=None,
                    help='using locally pre-trained model. The path of pre-trained model should be given')
parser.add_argument('--resume', type=str, metavar='PATH', default=None,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--cpu', action='store_true',
                    help='disable GPU training (default: False)')
parser.add_argument('--cpu-affinity', default=None, type=str,
                    help='CPU affinity, like "0xffff"')

# ========================== Important settings of training ==========================
# directory to the training dataset
parser.add_argument('--data-dir', type=str, required=True,
                    help='the path of dataset.')

# size of mini-batch
parser.add_argument('-b', '--batch-size', type=int, required=True, metavar='N',
                    help='mini-batch size')

# number of data loading workers
parser.add_argument('-j', '--workers', type=int, metavar='N', required=True,
                    help='number of data loading workers')

# training epochs
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of total epochs to run')

# compression ratio
parser.add_argument('--cr', metavar='N', type=int, default=4,
                    help='compression ratio')

# type of scheduler
parser.add_argument('--scheduler', type=str, default='const', choices=['const', 'cosine'],
                    help='learning rate scheduler')

# type of CSI feedback autoencoder
parser.add_argument('--name', type=str, default='CRNet',
                    help='model name')

# root directory to save training results and checkpoints
parser.add_argument('--root', type=str, default='./', help='checkpoint save root')

# method for online updating framework, currently only support the AO (alternating optimization) framework
parser.add_argument('--method', type=str, default='Alter',
                    help='scenario transferring method')

# sequence of scenario switching, the first one denoting the scenario on which the model is offline trained
parser.add_argument('--scenarios', type=str, default="CDADA",
                    help='transferring scenarios')

# flag denoting whether to train the model from scratch on the first scenario (offline training) 
parser.add_argument('--fresh', dest='fresh', action='store_true',
                    help='training the model from scratch')

# flag denoting whether to debug or release (influencing the result storage mode)
parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                    help='debug mode')

# ========================== Important parameter for the AO framework ==========================
# number of stored data in each scenario for knowledge review, i.e. the parameter n in the manuscript
parser.add_argument('--store-num', type=int, default=50, metavar='N',
                    help='number of saved samples')

# review period, i.e. the parameter p in the manuscript, denoting the number of scenario variations after which the knowledge review is conducted
parser.add_argument('--period', type=int, default=2, metavar='N',
                    help='review period')

# ========================== END ==========================
args = parser.parse_args()
