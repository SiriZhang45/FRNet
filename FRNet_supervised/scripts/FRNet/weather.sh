if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=720
model_name=MRMLP

root_path_name=./dataset/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2021
for pred_len in 720 
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 0 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 1 \
      --n_heads 8 \
      --d_model 64 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 50\
      --patience 9\
      --kernel_size 145\
      --lradj type3\
      --itr 1 --batch_size 128 --learning_rate 0.0004 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done