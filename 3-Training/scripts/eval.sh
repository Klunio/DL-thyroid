export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
--master_addr="127.0.0.2" \
--master_port=29501 \
--nproc_per_node=4 \
eval.py --init_method=env:// \
    --mag='10'\
    --model='inceptionv4'
ps aux | grep eval | awk '{print "kill -9 " $2}'| sh
