export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u /weatherIGM/src/trainer/train_miso.py \
 --config /weatherIGM/src/configs/global_forecast_model.yaml \
 --trainer.devices=[0,1,2,3,4,5,6,7] \
 --trainer.max_epochs=100 \