DATASET="cifar10"
CONF_FILE="configs/ft_cifar10_config.yaml"
OP_MDL_PTH="logs/model_dumps/$DATASET/erm"
# IP_MDL_PTH="logs/model_dumps/$DATASET/erm"
LG_PTH="logs/runs/$DATASET/erm"
PORT='12345'
N_REP=3
python3 main.py --log_path $LG_PTH --op_model_path $OP_MDL_PTH --train_mode finetune --conf $CONF_FILE --n_repeat $N_REP --port $PORT # --ip_model_path $IP_MDL_PTH