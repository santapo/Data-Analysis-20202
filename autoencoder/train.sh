python cli.py \
    --model.model_name='dummy' \
    --model.optimizer_name='sgd' \
    --model.lr=0.001 \
    --model.lr_scheduler_method_name='cosine' \
    --model.pretrained=true \
    --data.data_dir='data/' \
    --data.num_workers=2 \
    --data.batch_size=512 \
    --trainer.max_epochs=100 \
    --trainer.gpus=1 \
    --trainer.num_sanity_val_steps=2 \
    --trainer.default_root_dir='exps/dummy_bigger_sgd_cosine_512' \
    --pickle_embedd
