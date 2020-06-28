
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --data_dir ../data/ \
    --data_list ../data/train.csv \
    --batch_size 8 \
    --size 512 \
    --lr 1e-5 \
    --epoch 50



