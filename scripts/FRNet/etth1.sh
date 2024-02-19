if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
model_name=FRNet

root_path_name=./dataset/ETT-small
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021
for pred_len in 96
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_heads 1 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.05\
      --fc_dropout 0.1\
      --head_dropout 0.1\
      --patch_len 24\
      --stride 24\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --kernel_size 25\
      --lradj type4\
      --pred_head_type 'truncation'\
      --aggregation_type 'avg'\
      --channel_attention 0\
      --global_freq_pred 0\
      --period_list 24 48 72\
      --emb 96\
      --itr 1 --batch_size 128 --learning_rate 0.0003 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 192
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_heads 1 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.05\
      --fc_dropout 0.1\
      --head_dropout 0.1\
      --patch_len 24\
      --stride 24\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --kernel_size 25\
      --lradj type4\
      --pred_head_type 'truncation'\
      --aggregation_type 'avg'\
      --channel_attention 0\
      --global_freq_pred 0\
      --period_list 24 48\
      --emb 64\
      --itr 1 --batch_size 128 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 336
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_heads 1 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.05\
      --fc_dropout 0.1\
      --head_dropout 0.1\
      --patch_len 24\
      --stride 24\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --kernel_size 25\
      --lradj type4\
      --pred_head_type 'truncation'\
      --aggregation_type 'avg'\
      --channel_attention 0\
      --global_freq_pred 0\
      --period_list 24 48\
      --emb 64\
      --itr 1 --batch_size 128 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_heads 1 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.05\
      --fc_dropout 0.1\
      --head_dropout 0.1\
      --patch_len 24\
      --stride 24\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --kernel_size 25\
      --lradj type4\
      --pred_head_type 'truncation'\
      --aggregation_type 'linear'\
      --channel_attention 0\
      --global_freq_pred 1\
      --period_list 24 48 72\
      --emb 96\
      --itr 1 --batch_size 128 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done