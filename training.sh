PORT=9999
NAME='tless'
BATCHSIZE=64
MAX_EPOCHS=75
python SC6D_training.py  --ip_adress=127.0.0.1  --port=$PORT --nodes=1 --local_rank=0 --ngpus=2 --epochs=$MAX_EPOCHS --batchsize=$BATCHSIZE --dataset_name=$NAME