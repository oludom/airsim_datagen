import os
import argparse

rounds = 99
beta = 0.99
learning_rate = 0.001
learning_rate_change = 0.1
learning_rate_change_epoch = 10
loss_type = "MSE"
batch_size = 32
path_init_weight = "/media/data2/teamICRA/runs/Micha_weight/epoch11.pth"

for i in range(rounds):
    if i == 0 :
        os.system(f'python AirsimDAggerClient.py -w {path_init_weight} -t track{i} -b {beta}')
    else:
        model_weight_path = f'/media/data2/teamICRA/runs/ResNet32_ScaleV_body={batch_size}_lt={loss_type}_lr={learning_rate}_c={i-1}/best.pth'
        os.system(f'python AirsimDAggerClient.py -w {model_weight_path} -t track{i} -b {beta}')

    os.system(f'python3 ../train_newloader.py -pb /media/data2/teamICRA -db /media/data2/teamICRA/X4Gates_Circles_rl18tracks -n X1Gate100 -r {i}')
    beta -= 0.01




