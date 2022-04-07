#!/bin/bash

#skt/kobert는 아직 토크나이저를 만들지 않아서 넣으면 에러가 발생함.

python3 train.py \
    --model_name_or_path monologg/kobert\
    --config_name monologg/kobert\
    --tokenizer_name monologg/kobert\
    --train_file data/lawtime_passage.txt\
    --output_dir ./workspace\
    --num_train_epochs 9 \
    --per_device_train_batch_size 512 \
    --learning_rate 3e-5 \
    --max_seq_length 100 \
    --evaluation_strategy steps \
    --pooler_type avg \
    --mlp_only_train False\
    --temp 0.05 \
    --do_train True\
    --do_eval False\
    --fp16 True\
    --no_cuda False\
    --pre_evaluation False\
    --original_train_data_load False\
    --logging_strategy steps \
    --logging_steps=1 \
    --pad_to_max_length=False\
    --gradient_accumulation_steps=10\
    --save_steps=200\
    --logging_steps=1
    "$@"
