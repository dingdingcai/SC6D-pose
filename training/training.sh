PORT=9999
NAME='tless'
BATCHSIZE=64
BOP_DATASET = Path/to/dataset
python SC6D_training.py  --ip_adress=127.0.0.1  --port=$PORT --nodes=1 --local_rank=0 --ngpus=2 --rgb_dir=$BOP_DATASET --batchsize=$BATCHSIZE --dataset_name=$NAME