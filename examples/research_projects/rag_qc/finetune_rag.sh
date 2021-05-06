# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

# A sample finetuning run, you need to specify data_dir, output_dir and model_name_or_path
# run ./examples/rag/finetune_rag.sh --help to see all the possible options

python finetune_rag.py \
    --data_dir /nlu/users/giovanni_bonetta/transformers/examples/research_projects/rag_qc/prova_data \
    --output_dir /nlu/users/giovanni_bonetta/transformers/examples/research_projects/rag_qc/prova_finetune \
    --model_name_or_path facebook/rag-sequence-base \
    --model_type rag_sequence \
    --fp16 \
    --gpus 1 \
    --profile \
    --do_train \
    --do_predict \
    --n_val -1 \
    --train_batch_size 4 \
    --eval_batch_size 1 \
    --max_source_length 100 \
    --max_target_length 50 \
    --val_max_target_length 50 \
    --test_max_target_length 50 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --lr_scheduler polynomial \
    --learning_rate 3e-05 \
    --num_train_epochs 10 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 1 \
