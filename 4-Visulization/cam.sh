export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=4 python3 -m torch.distributed.launch \
--master_addr="127.0.0.2" \
--master_port=29503 \
--nproc_per_node=1 \
CAM.py --init_method=env://
    
ps aux | grep CAM | awk '{print "kill -9 " $2}'| sh
