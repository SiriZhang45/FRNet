if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
model_name=FRNet

root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

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
      --enc_in 321 \
      --e_layers 1 \
      --n_heads 8 \
      --d_model 64 \
      --d_ff 128 \
      --dropout 0.1\
      --fc_dropout 0.1\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --kernel_size 25\
      --lradj type3\
      --pred_head_type 'linear'\
      --aggregation_type 'avg'\
      --channel_attention 0\
      --global_freq_pred 1\
      --period_list 24 48 12\
      --emb 96\
      --itr 1 --batch_size 32 --learning_rate 0.0003 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --enc_in 321 \
      --e_layers 2 \
      --n_heads 8 \
      --d_model 64 \
      --d_ff 128 \
      --dropout 0.1\
      --fc_dropout 0.1\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --kernel_size 25\
      --lradj type1\
      --pred_head_type 'truncation'\
      --aggregation_type 'avg'\
      --channel_attention 0\
      --global_freq_pred 0\
      --period_list 24 48 12\
      --emb 96\
      --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --enc_in 321 \
      --e_layers 2 \
      --n_heads 8 \
      --d_model 64 \
      --d_ff 128 \
      --dropout 0.1\
      --fc_dropout 0.1\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --kernel_size 25\
      --lradj type3\
      --pred_head_type 'truncation'\
      --aggregation_type 'avg'\
      --channel_attention 0\
      --global_freq_pred 0\
      --period_list 24 48 12\
      --emb 96\
      --itr 1 --batch_size 32 --learning_rate 0.0003 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --enc_in 321 \
      --e_layers 2 \
      --n_heads 8 \
      --d_model 64 \
      --d_ff 128 \
      --dropout 0.1\
      --fc_dropout 0.1\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --kernel_size 25\
      --lradj type3\
      --pred_head_type 'truncation'\
      --aggregation_type 'linear'\
      --channel_attention 0\
      --global_freq_pred 0\
      --period_list 24 72 48 12\
      --emb 96\
      --itr 1 --batch_size 32 --learning_rate 0.0003 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done