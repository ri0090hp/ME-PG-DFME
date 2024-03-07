cd dfme;

lr_G_values=(8e-2 1e-3 5e-3 8e-3 1e-4 5e-4)
lr_S_values=(5e-2 8e-2 1e-2 1e-3 5e-3 8e-3) 
scale_values=(1e-1 3e-1 5e-1 7e-1 )
trained_generater=(MNIST FashionMNIST LDFractal Fractal) 
for lr_S in "${lr_S_values[@]}"; do
    for lr_G in "${lr_G_values[@]}"; do
        for scale in "${scale_values[@]}"; do
            python3 train.py --dataset FashionMNIST --medflag octmnist --target_load_name AttackerNetworkSmall_MNIST_epoch_30_data_FashionMNIST --G_net_load_path 0 --G_net_data FashionMNIST --device 0 --grad_m 1 --batch_size 256 --scale "$scale" --steps 0.1 0.3 0.5 --query_budget 2 --epoch_itrs 20 --g_iter 1 --d_iter 5 --lr_S "$lr_S" --lr_G "$lr_G" --nz 256 --log_dir /workspace/DFNE_log/ --student_model resnet18_8x --loss l1 --weight_decay 0.0005 --momentum 0.9 --seed 19515 --approx_grad 1 --grad_m 1 --grad_epsilon 0.001 --forward_differences 1 --no_logits 1 --logit_correction "mean" --rec_grad_norm 0 --MAZE 0 --store_checkpoints 1;
        done
    done
done
#実験結果のプロットは/workspace/GitDFME/dfme/Original_Accuracy_plot.ipynb
