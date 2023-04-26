comment=2022-03-21_baseline

rm -rf ../checkpoints/${comment}
rm -rf ../runs/${comment}

mkdir ../checkpoints/${comment}
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
--master_addr="127.0.0.2" \
--master_port=29501 \
--nproc_per_node=8 \
train_mil_attention.py --epochs=150\
    --mag='10'\
    --lr='0.0002'\
    --lrdrop=50\
    --padding=32\
    --model='inceptionv4'\
    --weight-decay='5e-3'\
    --momentum='0.9'\
    --pretrain\
    --init_method=env:// \
    --comment=${comment}

sleep 3s
ps aux | grep train_mil | awk '{print "kill -9 " $2}'| sh
