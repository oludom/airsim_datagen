import os
import argparse

rounds = 5
beta = 0.9
learning_rate = 0.001
learning_rate_change = 0.1
learning_rate_change_epoch = 10
loss_type = "MSE"
batch_size = 32
path_init_weight = "/media/data2/teamICRA/runs/Micha_weight/epoch11.pth"
for i in range(rounds):
    """
    Step :
    Load Init_weight => DAggerClient
    Run DAggerClient => Track0
    Train track 0 => Weight run(0)
    2ND LOOP :
        Load last Weight : i - 1 
        Run DAgger => new data
        Save new data => Train => New_weight
        LOOP => i = rounds

    """
    if i == 0 :
        os.system(f'python test_DAgger.py -w {path_init_weight} -t track{i}')
    else:
        model_weight_path = f'/media/data2/teamICRA/runs/ResNet32_ScaleV_body={batch_size}_lt={loss_type}_lr={learning_rate}_c={i-1}/best.pth'
        os.system(f'python test_DAgger.py -w {model_weight_path} -t track{i}')

    os.system(f'python3 ../train_newloader.py -pb /media/data2/teamICRA -db /media/data2/teamICRA/X4Gates_Circles_rl18tracks -n X1Gate100 -r {i}')
    beta -= 0.2




