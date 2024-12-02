
python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/new_data6 \
      --data_path X.csv \
      --model_id new_rehab_wb6 \
      --model InsFormer \
      --data rehab \
      --itr 1 \
      --train_epochs 100 \
      --lradj 'constant' \
      --seq_len 7 \
      --pred_len 7 \
      --batch_size 4 \
      --patch_len 7 \
      --stride 1 \
      --label_len 3 \
      --moving_avg 1 \
      --learning_rate 0.0001
