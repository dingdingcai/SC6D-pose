PORT=9999
NAME='tless'
BATCHSIZE=64
python SC6D_training.py  --ip_adress=127.0.0.1  --port=$PORT --nodes=1 --local_rank=0 --ngpus=2 --batchsize=$BATCHSIZE --dataset_name=$NAME