from time import time
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data
import torch.nn as nn
from utils.file_utils import check_folders
from batchgenerators.utilities.file_and_folder_operations import *
from models.unet_ccsdg import UNetCCSDG, Projector
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list
from datasets.utils.transform import collate_fn_tr_styleaug, collate_fn_ts
from utils.lr import adjust_learning_rate
from utils.metrics.dice import get_hard_dice
from torchvision.utils import make_grid
import torch.nn.functional as F
import torchvision
from datasets.dataloaders.PROSTATE_dataloader import PROSTATE_dataset
from config import Seg_loss,weighted_Seg_loss
from datasets.utils.transform import collate_prostate_styleaug, collate_prostate_test
from utils.lr import adjust_learning_rate,adjust_weight,adjust_weight_new
from test_PROSTATE import Test
from models.unet_mixnoise import UNet_Mix,generater
from utils.loss import fourier_exchage_loss
from PIL import Image

def gradient(img):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_x = sobel_x.view(1, 1, 3, 3)

    gradient_img = torch.zeros_like(img)
    for channel in range(img.size(1)):
        channel_img = img[:,channel:channel+1].float()
        sobel_x = sobel_x.to(channel_img.device)
        channel_gradient = F.conv2d(channel_img, sobel_x, padding=1)
        gradient_img[:,channel:channel+1] = channel_gradient

    return gradient_img

def weight_map(a,b,c,threshold=0.5,weight=1):
    # a∩c - a∩b∩c
    zero = torch.zeros_like(a)
    one = torch.ones_like(a)
    a = torch.sigmoid(a)
    b = torch.sigmoid(b)
    c = torch.sigmoid(c)
    a = torch.where(a > threshold, one, zero)
    b = torch.where(b > threshold, one, zero)
    c = torch.where(c > threshold, one, zero)

    intersection_ac = torch.logical_and(a, c)
    intersection_abc = torch.logical_and(torch.logical_and(a, b), c)
    result_tensor = torch.logical_and(intersection_ac, torch.logical_not(intersection_abc))
    weight_map = torch.ones_like(result_tensor, dtype=torch.float32)
    weight_map[result_tensor] = weight
    return weight_map

class EpochLR(torch.optim.lr_scheduler._LRScheduler):
    # lr_n = lr_0 * (1 - epoch / epoch_nums)^gamma
    def __init__(self, optimizer, epochs, gamma=0.9, last_epoch=-1):
        self.lr = optimizer.param_groups[0]['lr']
        self.epochs = epochs
        self.gamma = gamma
        super(EpochLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.lr * pow((1. - (self.last_epoch + 1) / self.epochs), self.gamma)]

@torch.no_grad()
def uncertainty_estimation(model, x, sigma=0.05, num_samples=5):
    uncertainties = []
    for i in range(num_samples):
        noisy_input = x + torch.randn_like(x) * sigma
        output = model(noisy_input)
        uncertainty = torch.sigmoid(output)
        # uncertainty = F.softmax(output)
        uncertainties.append(uncertainty)

    uncertainty = torch.stack(uncertainties).mean(dim=0)
    entropy = -torch.sum(uncertainty * torch.log(uncertainty + 1e-10), dim=1)
    # entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min())
    min_value = 0.8
    max_value = 1.2
    entropy = min_value + (entropy - entropy.min()) * (max_value - min_value) / (
                entropy.max() - entropy.min())

    return entropy


def train(args):
    seg_cost = Seg_loss()
    weight_seg_cost = weighted_Seg_loss()
    model_name = args.model
    gpu = tuple(args.gpu)
    log_folder = args.log_folder
    tag = args.tag
    log_folder = join(log_folder, model_name+'_'+tag)
    patch_size = tuple(args.patch_size)
    batch_size = args.batch_size
    initial_lr = args.initial_lr
    save_interval = args.save_interval
    num_epochs = args.num_epochs
    continue_training = args.continue_training
    num_threads = args.num_threads
    root_folder = args.root
    tr_csv = tuple(args.tr_csv)
    ts_csv = tuple(args.ts_csv)
    shuffle = not args.no_shuffle
    tau = 0.1
    fourier_aug = args.fourier_aug

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu])
    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    writer = SummaryWriter(log_dir=tensorboard_folder)

    source_name = ['I2CVB.csv']

    tr_img_list, tr_label_list = convert_labeled_list(args.root, source_name)
    # ts_img_list, ts_label_list = convert_labeled_list(args.root, test_csv)

    # tr_img_list, tr_label_list = convert_labeled_list(tr_csv)
    tr_dataset = PROSTATE_dataset(root_folder, tr_img_list, tr_label_list, patch_size, img_normalize=False)
    # ts_dataset = PROSTATE_dataset(root_folder, ts_img_list, ts_label_list, patch_size)
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                drop_last=True,
                                                collate_fn=collate_prostate_styleaug)
    base1_dataloader = None
    base2_dataloader = None
    base3_dataloader = None
    base4_dataloader = None
    base5_dataloader = None
    
    if(args.domain_test):
            # base1
            base1_csv = ['UCL.csv']
            #base1_csv = ['BMC.csv']
            base1_img_list, base1_label_list = convert_labeled_list(args.root, base1_csv)
            base1_dataset = PROSTATE_dataset(root_folder, base1_img_list, base1_label_list,patch_size, img_normalize=True)
            base1_dataloader = torch.utils.data.DataLoader(dataset=base1_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True,
                                         collate_fn=collate_prostate_test, )
            # base2
            base2_csv = ['UCL.csv']
            base2_img_list, base2_label_list = convert_labeled_list(args.root, base2_csv)
            base2_dataset = PROSTATE_dataset(root_folder, base2_img_list, base2_label_list,patch_size, img_normalize=True)
            base2_dataloader = torch.utils.data.DataLoader(dataset=base2_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True,
                                         collate_fn=collate_prostate_test, )

            # base3
            base3_csv = ['BIDMC.csv']
            base3_img_list, base3_label_list = convert_labeled_list(args.root, base3_csv)
            base3_dataset = PROSTATE_dataset(root_folder, base3_img_list, base3_label_list,patch_size, img_normalize=True)
            base3_dataloader = torch.utils.data.DataLoader(dataset=base3_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True,
                                         collate_fn=collate_prostate_test, )

            # base4
            base4_csv = ['RUNMC.csv']
            base4_img_list, base4_label_list = convert_labeled_list(args.root, base4_csv)
            base4_dataset = PROSTATE_dataset(root_folder, base4_img_list, base4_label_list,patch_size, img_normalize=True)
            base4_dataloader = torch.utils.data.DataLoader(dataset=base4_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True,
                                         collate_fn=collate_prostate_test, )

            # base5
            base5_csv = ['HK.csv']
            base5_img_list, base5_label_list = convert_labeled_list(args.root, base5_csv)
            base5_dataset = PROSTATE_dataset(root_folder, base5_img_list, base5_label_list,patch_size, img_normalize=True)
            base5_dataloader = torch.utils.data.DataLoader(dataset=base5_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True,
                                         collate_fn=collate_prostate_test, )



    model = UNetCCSDG(num_classes=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    projector = Projector()
    projector.to(device)

    Generater = generater(mixstyle_layers=args.mixstyle_layers,batch=4)
    Generater.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.99, nesterov=True)
    optimizer_prompt = torch.optim.SGD(list(projector.parameters()) + [model.channel_prompt],
                                       lr=initial_lr, momentum=0.99, nesterov=True)
    optimizer_generater = torch.optim.SGD(Generater.parameters(), lr=initial_lr, momentum=0.99, nesterov=True)
    optimizer_noise = torch.optim.SGD(list(Generater.noiseEncoder.parameters()), lr=0.1, momentum=0.99, nesterov=True)


    schedulerA = EpochLR(optimizer, epochs=101, gamma=0.9)
    schedulerB = EpochLR(optimizer_prompt, epochs=101, gamma=0.9)
    schedulerC = EpochLR(optimizer_generater, epochs=101, gamma=0.9)
    schedulerD = EpochLR(optimizer_noise, epochs=101, gamma=0.9)


    start_epoch = 0
    if continue_training:
        assert isfile(join(model_folder, 'model_latest.model')), 'missing model checkpoint!'
        params = torch.load(join(model_folder, 'model_latest.model'))
        model.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        start_epoch = params['epoch']
    print('start epoch: {}'.format(start_epoch))

    amp_grad_scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    start = time()
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}:'.format(epoch))
        start_epoch = time()
        model.train()
        lr = adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs)
        print('  lr: {}'.format(lr))

        train_loss_list = list()
        train_disc_dice_list = list()
        train_cup_dice_list = list()
        content_loss_list = list()
        style_loss_list = list()
        for iter, batch in enumerate(tr_dataloader):
            name = batch['name']
            data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
            fda_data = torch.from_numpy(batch['fda_data']).cuda().to(dtype=torch.float32)
            GLA_data = torch.from_numpy(batch['GLA']).cuda().to(dtype=torch.float32)
            seg = torch.from_numpy(batch['mask']).cuda().to(dtype=torch.float32)
            clean_data = torch.from_numpy(batch['clean_data']).cuda().to(dtype=torch.float32)

            optimizer.zero_grad()
            with autocast():
                output = model(fda_data, tau=tau)
                loss = seg_cost(output, seg)
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()

            optimizer.zero_grad()
            with autocast():
                output = model(GLA_data, tau=tau)
                loss = seg_cost(output, seg)
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()


            optimizer.zero_grad()
            with autocast():
                output_seg = model(data, tau=tau)
                loss = seg_cost(output_seg, seg)
                # loss = criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + \
                #        criterion(output[:, 1], (seg[:, 0] == 2) * 1.0)
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()

            train_loss_list.append(loss.detach().cpu().numpy())

            # output_sigmoid = torch.sigmoid(output)
            # train_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
            # train_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))

            if (fourier_aug):
                # noise training
                l_n = adjust_weight_new(epoch, 5e-3)
                l_f = adjust_weight_new(epoch, 5e-5)

                optimizer_noise.zero_grad()
                x, sfs = model.encoder(data)
                recon_img = Generater(x, sfs, data)
                # _, recon_img = fourier_exchage_loss(recon_img, data, l_w=l_f, l_l=1e-2)
                output = model(recon_img.data)
                seg_loss_n = -seg_cost(output, seg)
                seg_loss_n.backward()
                optimizer_noise.step()
                
                
                # ----------------------------------------------------------------

                optimizer_noise.zero_grad()
                x, sfs = model.encoder(data)
                recon_img = Generater(x, sfs, data)
                output = model(recon_img.data)
                seg_loss_n = -seg_cost(output, seg)
                seg_loss_n.backward()
                optimizer_noise.step()

                optimizer_noise.zero_grad()
                x, sfs = model.encoder(data)
                recon_img = Generater(x, sfs, data)
                output = model(recon_img.data)
                seg_loss_n = -seg_cost(output, seg)
                seg_loss_n.backward()
                optimizer_noise.step()
                
                optimizer_noise.zero_grad()
                x, sfs = model.encoder(data)
                recon_img = Generater(x, sfs, data)
                output = model(recon_img.data)
                seg_loss_n = -seg_cost(output, seg)
                seg_loss_n.backward()
                optimizer_noise.step()

                # ----------------------------------------------------------------
                

                # generate new img
                optimizer_generater.zero_grad()
                # x, sfs = model.encoder(data)
                recon_img = Generater(x, sfs, data)
                #Fourier_loss, recon_img = fourier_exchage_loss(output.data, data, l_w=l_f, l_l=1e-2)
                mse_loss = nn.MSELoss()(recon_img, data)
                #recon_loss = Fourier_loss + l_n * mse_loss
                recon_loss = l_n * mse_loss
                recon_loss.backward()
                optimizer_generater.step()
                
                
                # standard traing
                optimizer.zero_grad()
                output_noise = model(recon_img.data)

                # uncertainty
                # uncertainty = uncertainty_estimation(model, data)
                # more_attention = weight_map(output_seg, output_noise, seg, weight=1.2).squeeze()
                # less_attention = weight_map(output_seg, seg, output_noise, weight=0.8).squeeze()
                # uncertainty = uncertainty * more_attention * less_attention
                # loss = weight_seg_cost(output_noise, seg, uncertainty)
                loss = seg_cost(output_noise, seg)
                loss.backward()
                optimizer.step()
            
            
            name_list = name[0].split('/')
            clean_data = clean_data.detach()[0].cpu().numpy()
            clean_data_img = Image.fromarray(clean_data.transpose(1, 2, 0).astype(np.uint8))
            clean_data_img.save('save_img/' + name_list[-2] + '/' + name_list[-1] + '.png')
              

            recon = recon_img.detach()[0].cpu().numpy()
            
            
            for b in range(recon.shape[0]):
                for c in range(recon.shape[1]):
                    std=clean_data[b, c].std()
                    mean=clean_data[b, c].mean()
                    recon[b, c] = recon[b, c] * std + mean  
                  
                  
            recon_img = Image.fromarray(recon.transpose(1, 2, 0).astype(np.uint8))
            recon_img.save('save_img1/' + name_list[-2] + '/' + name_list[-1] + '_rec.png')
            
            
            output_oc = torch.sigmoid(output_noise[0, 0])
            zero = torch.zeros_like(output_oc)
            one = torch.ones_like(output_oc)
            output_oc = torch.where(output_oc > 0.5, one, zero)

            oc = output_oc.detach().cpu().numpy()
            segg = np.uint8(oc * 255)
            image = Image.fromarray(segg, 'L')
            image.save('save_img1/' + name_list[-2] + '/' + name_list[-1] + '_seg.png')

            mask = seg[0, 0].cpu().numpy()
            segg = np.uint8((mask > 0) * 255)
            image = Image.fromarray(segg, 'L')
            image.save('save_img1/' + name_list[-2] + '/' + name_list[-1] + '_gt.png')
            


            del seg

        schedulerA.step()
        schedulerB.step()
        schedulerC.step()
        schedulerD.step()

        mean_tr_loss = np.mean(train_loss_list)
        mean_content_loss = np.mean(content_loss_list)
        mean_style_loss = np.mean(style_loss_list)
        mean_tr_disc_dice = np.mean(train_disc_dice_list)
        mean_tr_cup_dice = np.mean(train_cup_dice_list)
        writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
        writer.add_scalar("Train Scalars/Train Content Loss", mean_content_loss, epoch)
        writer.add_scalar("Train Scalars/Train Style Loss", mean_style_loss, epoch)
        writer.add_scalar("Train Scalars/Disc Dice", mean_tr_disc_dice, epoch)
        writer.add_scalar("Train Scalars/Cup Dice", mean_tr_cup_dice, epoch)
        print('  Tr loss: {}\n'
              '  Tr disc dice: {}; Cup dice: {}'.format(mean_tr_loss, mean_tr_disc_dice, mean_tr_cup_dice))



        if epoch % args.valid_frequency == 0 and base1_dataloader is not None:
            # # source
            # test = Test(config=args, test_loader=tr_dataloader,name='source',model=model)
            # result_list = test.test()

            # base1
            test = Test(config=args, test_loader=base1_dataloader,name='BIDMC',model=model)
            result_list,dice1 = test.test()

            # base2
            test = Test(config=args, test_loader=base2_dataloader,name='HK',model=model)
            result_list,dice2 = test.test()

            # base3
            test = Test(config=args, test_loader=base3_dataloader,name='UCL',model=model)
            result_list,dice3 = test.test()

            # base4
            test = Test(config=args, test_loader=base4_dataloader,name='BMC',model=model)
            result_list,dice4 = test.test()

            # base5
            test = Test(config=args, test_loader=base5_dataloader,name='RUNMC',model=model)
            result_list,dice5 = test.test()
            
                
                
                
                


            train_loss_list.append(loss.detach().cpu().numpy())
            
            

            num1 = len(base1_dataloader)
            num2 = len(base2_dataloader)
            num3 = len(base3_dataloader)
            num4 = len(base4_dataloader)
            num5 = len(base5_dataloader)
            dice = (num1*dice1+num2*dice2+num3*dice3+num4*dice4+num5*dice5)/(num1+num2+num3+num4+num5)
            print('dice all',dice)

        if epoch % save_interval == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format('latest'))
            torch.save(saved_model, join(model_folder, 'model_latest.model'))

        if (epoch+1) % 200 == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format(epoch+1))
            torch.save(saved_model, join(model_folder, 'model_{}.model'.format(epoch+1)))

        time_per_epoch = time() - start_epoch
        print('  Durations: {}'.format(time_per_epoch))
        writer.add_scalar("Time/Time per epoch", time_per_epoch, epoch)
    saved_model = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    print('Saving model_{}.model...'.format('final'))
    torch.save(saved_model, join(model_folder, 'model_final.model'))
    if isfile(join(model_folder, 'model_latest.model')):
        os.remove(join(model_folder, 'model_latest.model'))
    total_time = time() - start
    print("Running %d epochs took a total of %.2f seconds." % (num_epochs, total_time))

    # inference
    from inference.inference_nets.inference_unet_ccsdg import inference
    for ts_csv_path in ts_csv:
        inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
        print("Running inference: {}".format(inference_tag))
        inference('model_final.model', gpu, log_folder, patch_size, root_folder, [ts_csv_path], inference_tag, tau=tau)

