CUDA_VISIBLE_DEVICES=1 python3 main.py \
    --do_predict \
    --batch_size 4 \
    --size 512 \
    --data_dir ../data/ \
    --load_model ../model/model_1 \
    --predict_file ../data/dev.csv \
    --output prediction.csv


    



