import os
import argparse
from utils.file_utils import gen_random_str

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="unet_ccsdg", required=False, help='Model name.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[3], required=False, help='Device id.')
    parser.add_argument('--log_folder', default='saved', required=False, help='Log folder.')
    parser.add_argument('--tag', default='source_BinRushed', required=False, help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False, help='patch size.')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size.')
    parser.add_argument('--initial_lr', type=float, default=1e-1, required=False, help='initial learning rate.')
    parser.add_argument('--save_interval', type=int, default=50, required=False, help='save_interval.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true', help="restore from checkpoint and continue training.")
    parser.add_argument('--no_shuffle', default=False, required=False, action='store_true', help="No shuffle training set.")
    parser.add_argument('--num_threads', type=int, default=8, required=False, help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='RIGAPlus', required=False, help='dataset root folder.')
    parser.add_argument('--tr_csv', nargs='+', default=['RIGAPlus/Magrabia_train.csv'], required=False, help='training csv file.')
    parser.add_argument('--ts_csv', nargs='+', default=['RIGAPlus/Magrabia_test.csv'], required=False, help='test csv file.')
    parser.add_argument('--num_epochs', type=int, default=101, required=False, help='num_epochs.')
    parser.add_argument('--valid_frequency', type=int, default=20, required=False, help='num_epochs.')
    parser.add_argument('--domain_test', default=True, required=False, action='store_true', help="")
    parser.add_argument('--mixstyle_layers', default=['layer2','layer3'], required=False, help='Log folder.')
    parser.add_argument('--random_type', default='MixNoise', required=False, help='Log folder.')
    parser.add_argument('--Source_Dataset', nargs='+', type=str, default=['BinRushed'],help='BinRushed/Magrabia/REFUGE/ORIGA/Drishti_GS')
    parser.add_argument('--dataset', default='RIGA', required=False, help='Log folder.')
    parser.add_argument('--fourier_aug', default=True, required=False, help='Log folder.')

    args = parser.parse_args()
    return args

def main():
    args = parser()
    model_name = args.model
    if model_name == 'unet':
        from training.train_nets.train_unet import train
    elif model_name == 'unet_ccsdg':
        if(args.dataset == 'RIGA'):
            from training.train_nets.train_ocod_fourier import train
        elif(args.dataset == 'PROSTATE'):
            from training.train_nets.train_ocod_fourier import train
            args.root = 'Processed_data_nii'
            args.patch_size = [384, 384]
            args.Source_Dataset = ['BMC.csv']
        else:
            print('No dataset named "{}"!'.format(args.dataset))
            return
    else:
        print('No model named "{}"!'.format(model_name))
        return
    train(args)

if __name__ == '__main__':
    main()