from time import time
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
import torch
import numpy as np
import torch.nn as nn
from utils.file_utils import check_folders
from batchgenerators.utilities.file_and_folder_operations import *
from models.unet_ccsdg import UNetCCSDG, Projector
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list
from datasets.utils.transform import collate_fn_tr_styleaug, collate_fn_ts
from utils.lr import adjust_learning_rate,adjust_weight,adjust_weight_new
from utils.metrics.dice import get_hard_dice
from torchvision.utils import make_grid
import torch.nn.functional as F
import torchvision
from config import Seg_loss,weighted_Seg_loss
from models.unet_mixnoise import UNet_Mix,generater
from utils.loss import fourier_exchage_loss
from datasets.utils.saliency_balancing_fusion import get_SBF_map
from torch.autograd import Variable
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

# @torch.no_grad()
# def uncertainty_estimation(model, x, sigma=0.05, num_samples=5):
#     uncertainties = []
#     for i in range(num_samples):
#         noisy_input = x + torch.randn_like(x) * sigma
#         output = model(noisy_input)
#         uncertainty = torch.sigmoid(output)
#         # uncertainty = F.softmax(output)
#         uncertainties.append(uncertainty)
#
#     uncertainty = torch.stack(uncertainties).mean(dim=0)
#     print('uncertain',uncertainty.size())
#     entropy = -torch.sum(uncertainty * torch.log(uncertainty + 1e-10), dim=1)
#     print('entroy',entropy.size())
#     # entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min())
#     min_value = 0.8
#     max_value = 1.2
#     entropy = min_value + (entropy - entropy.min()) * (max_value - min_value) / (
#                 entropy.max() - entropy.min())
#
#     print('ent',entropy.size())
#
#     return entropy

@torch.no_grad()
def uncertainty_estimation(model, x, sigma=0.05, num_samples=5):
    disc_uncertainties = []
    cup_uncertainties = []

    for i in range(num_samples):
        noisy_input = x + torch.randn_like(x) * sigma
        output = model(noisy_input)
        softmax_output = torch.softmax(output, dim=1)

        disc_probabilities = softmax_output[:, 0, :, :].unsqueeze(1)
        cup_probabilities = softmax_output[:, 1, :, :].unsqueeze(1)
        disc_uncertainties.append(disc_probabilities)
        cup_uncertainties.append(cup_probabilities)

    disc_uncertainty = torch.stack(disc_uncertainties).mean(dim=0)
    cup_uncertainty = torch.stack(cup_uncertainties).mean(dim=0)
    disc_entropy = -torch.sum(disc_uncertainty * torch.log(disc_uncertainty + 1e-10), dim=1)
    cup_entropy = -torch.sum(cup_uncertainty * torch.log(cup_uncertainty + 1e-10), dim=1)

    min_value = 0.8
    max_value = 1.2
    disc_entropy = min_value + (disc_entropy - disc_entropy.min()) * (max_value - min_value) / (
            disc_entropy.max() - disc_entropy.min())
    cup_entropy = min_value + (cup_entropy - cup_entropy.min()) * (max_value - min_value) / (
            cup_entropy.max() - cup_entropy.min())

    class_entropy = torch.cat([disc_entropy.unsqueeze(1),cup_entropy.unsqueeze(1)],dim=1)

    return class_entropy


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

    source_name = args.Source_Dataset
    source_csv = []
    test_csv = []
    for s_n in source_name:
        source_csv.append(s_n + '_train.csv')
        test_csv.append(s_n + '_test.csv')


    tr_img_list, tr_label_list = convert_labeled_list(args.root, source_csv)
    ts_img_list, ts_label_list = convert_labeled_list(args.root, test_csv)

    # tr_img_list, tr_label_list = convert_labeled_list(tr_csv)
    tr_dataset = RIGA_labeled_set(root_folder, tr_img_list, tr_label_list, patch_size, img_normalize=False)
    ts_dataset = RIGA_labeled_set(root_folder, ts_img_list, ts_label_list, patch_size)
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                collate_fn=collate_fn_tr_styleaug)
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads//2,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=collate_fn_ts)

    base1_dataloader = None
    base2_dataloader = None
    base3_dataloader = None
    if(args.domain_test):
            # base1
            base1_csv = ['MESSIDOR_Base1_test.csv']
            base1_img_list, base1_label_list = convert_labeled_list(args.root, base1_csv)
            base1_dataset = RIGA_labeled_set(root_folder, base1_img_list, base1_label_list,patch_size, img_normalize=True)
            base1_dataloader = torch.utils.data.DataLoader(dataset=base1_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True,
                                         collate_fn=collate_fn_ts, )
            # base2
            base2_csv = ['MESSIDOR_Base2_test.csv']
            base2_img_list, base2_label_list = convert_labeled_list(args.root, base2_csv)
            base2_dataset = RIGA_labeled_set(root_folder, base2_img_list, base2_label_list,patch_size, img_normalize=True)
            base2_dataloader = torch.utils.data.DataLoader(dataset=base2_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True,
                                         collate_fn=collate_fn_ts, )
            # base3
            base3_csv = ['MESSIDOR_Base3_test.csv']
            base3_img_list, base3_label_list = convert_labeled_list(args.root, base3_csv)
            base3_dataset = RIGA_labeled_set(root_folder, base3_img_list, base3_label_list,patch_size, img_normalize=True)
            base3_dataloader = torch.utils.data.DataLoader(dataset=base3_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True,
                                         collate_fn=collate_fn_ts, )

    model = UNetCCSDG(num_classes=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    projector = Projector()
    projector.to(device)

    Generater = generater(mixstyle_layers=args.mixstyle_layers,batch=args.batch_size)
    Generater.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.99, nesterov=True)
    optimizer_prompt = torch.optim.SGD(list(projector.parameters()) + [model.channel_prompt],
                                       lr=initial_lr, momentum=0.99, nesterov=True)
    optimizer_generater = torch.optim.SGD(Generater.parameters(), lr=initial_lr, momentum=0.99, nesterov=True)
    optimizer_noise = torch.optim.SGD(list(Generater.noiseEncoder.parameters()), lr=0.01, momentum=0.99, nesterov=True)

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
            clean_data = torch.from_numpy(batch['clean_data']).cuda().to(dtype=torch.float32)
            data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
            fda_data = torch.from_numpy(batch['fda_data']).cuda().to(dtype=torch.float32)
            GLA_data = torch.from_numpy(batch['GLA']).cuda().to(dtype=torch.float32)
            LLA_data = torch.from_numpy(batch['LLA']).cuda().to(dtype=torch.float32)
            seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
            
            optimizer.zero_grad()
            with autocast():
                output_seg = model(fda_data, tau=tau)
                loss = criterion(output_seg[:, 0], (seg[:, 0] > 0) * 1.0) + \
                       criterion(output_seg[:, 1], (seg[:, 0] == 2) * 1.0)
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()

            optimizer.zero_grad()
            with autocast():
                output_seg = model(data, tau=tau)
                loss = criterion(output_seg[:, 0], (seg[:, 0] > 0) * 1.0) + \
                       criterion(output_seg[:, 1], (seg[:, 0] == 2) * 1.0)
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()

            train_loss_list.append(loss.detach().cpu().numpy())


            if(fourier_aug):
                l_n = adjust_weight_new(epoch, 5e-3)
                l_f = adjust_weight_new(epoch, 5e-4)

                # noise training
                optimizer_noise.zero_grad()
                x, sfs = model.encoder(data)
                recon_img = Generater(x, sfs, data)
                _, recon_img = fourier_exchage_loss(recon_img, data)
                output = model(recon_img.detach())
                seg_loss_n = -criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) - \
                criterion(output[:, 1], (seg[:, 0] == 2) * 1.0)

                seg_loss_n.backward()
                optimizer_noise.step()

                # generate new img
                optimizer_generater.zero_grad()
                # x, sfs = model.encoder(data)
                output = Generater(x, sfs, data)
                #recon_img= Generater(x, sfs, data)
                Fourier_loss, recon_img = fourier_exchage_loss(output.detach(), data, l_w=l_f, l_l=1e-1)
                mse_loss = nn.MSELoss()(recon_img, data)
                recon_loss = Fourier_loss + l_n * mse_loss
                #recon_loss = mse_loss
                
                
                recon_loss.requires_grad_()
                recon_loss.backward()
                optimizer_generater.step()

                # standard traing
                optimizer.zero_grad()
                output_noise = model(recon_img.detach())

                disc_noise = output_noise[:, 0, :, :].unsqueeze(1)
                cup_noise = output_noise[:, 1, :, :].unsqueeze(1)
                disc_output = output_seg[:, 0, :, :].unsqueeze(1)
                cup_output = output_seg[:, 1, :, :].unsqueeze(1)
                disc_seg = (seg[:, 0] > 0).float().unsqueeze(1)
                cup_seg = (seg[:, 0] == 2).float().unsqueeze(1)

                # uncertainty
                uncertainty = uncertainty_estimation(model, data)
                more_attention_disc = weight_map(disc_output, disc_noise, disc_seg, weight=1.2)
                more_attention_cup = weight_map(cup_output, cup_seg, cup_noise, weight=1.2)
                less_attention_disc = weight_map(disc_output, disc_seg, disc_noise, weight=0.8)
                less_attention_cup = weight_map(cup_output, cup_seg, cup_noise, weight=0.8)

                uncertainty_od = uncertainty[0, 0].cpu().detach().numpy()
                uncertainty_oc = uncertainty[0, 1].cpu().detach().numpy()

                min_value = 0.8
                max_value = 1.2
                normalized_data = ((uncertainty_oc - min_value) / (max_value - min_value)) * 255
                normalized_data = np.clip(normalized_data, 0, 255)
                normalized_data = normalized_data.astype(np.uint8)

                disc_uncertainty = uncertainty[:, 0, :, :].unsqueeze(1) * more_attention_disc * less_attention_disc
                cup_uncertainty = uncertainty[:, 1, :, :].unsqueeze(1) * more_attention_cup * less_attention_cup

                loss_disc = weight_seg_cost(output_noise[:, 0, :, :].unsqueeze(1), (seg[:, 0] > 0).float().unsqueeze(1),
                                            disc_uncertainty)
                loss_cup = weight_seg_cost(output_noise[:, 1, :, :].unsqueeze(1), (seg[:, 0] == 2).float().unsqueeze(1),
                                           cup_uncertainty)
                loss = loss_disc + loss_cup
        
            
            name_list = name[0].split('/')
            recon = recon_img.detach()[0].cpu().numpy()
            clean_data = clean_data.detach()[0].cpu().numpy()
            for b in range(recon.shape[0]):
                for c in range(recon.shape[1]):
                    std=clean_data[b, c].std()
                    mean=clean_data[b, c].mean()
                    recon[b, c] = recon[b, c] * std + mean
            
            
            
            recon_img = Image.fromarray(recon.transpose(1, 2, 0).astype(np.uint8))
            recon_img.save('save_img/' + name_list[-2] + '/' + name_list[-1] + '_rec.png')
            

            output_sigmoid = torch.sigmoid(output_seg)
            train_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
            train_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
            del seg

        schedulerA.step()
        schedulerB.step()
        if(fourier_aug):
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

        val_loss_list = list()
        val_disc_dice_list = list()
        val_cup_dice_list = list()

        if epoch % args.valid_frequency == 0 and ts_dataloader is not None:
            with torch.no_grad():
                model.eval()
                for iter, batch in enumerate(ts_dataloader):
                    data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                    seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
                    with autocast():
                        output = model(data, tau=tau)
                        loss = criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + criterion(output[:, 1],
                                                                                          (seg[:, 0] == 2) * 1.0)
                    val_loss_list.append(loss.detach().cpu().numpy())
                    output_sigmoid = torch.sigmoid(output)
                    val_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
                    val_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))

            mean_val_loss = np.mean(val_loss_list)
            mean_val_disc_dice = np.mean(val_disc_dice_list)
            mean_val_cup_dice = np.mean(val_cup_dice_list)
            writer.add_scalar("Val Scalars/Val Loss", mean_val_loss, epoch)
            writer.add_scalar("Val Scalars/Disc Dice", mean_val_disc_dice, epoch)
            writer.add_scalar("Val Scalars/Cup Dice", mean_val_cup_dice, epoch)
            writer.add_image('Val/Input', make_grid(data[:10], 10, normalize=True), epoch)
            writer.add_image('Val/Output Disc', make_grid(output_sigmoid[:10, 0][:, np.newaxis], 10, normalize=True), epoch)
            writer.add_image('Val/Output Cup', make_grid(output_sigmoid[:10, 1][:, np.newaxis], 10, normalize=True), epoch)
            writer.add_image('Val/Seg', make_grid(seg[:10], 10, normalize=True), epoch)

            print('  Val loss: {}\n'
                '  Val disc dice: {}; Cup dice: {}'.format(mean_val_loss, mean_val_disc_dice, mean_val_cup_dice))

            if (args.domain_test):
                base1_loss_list = list()
                base1_disc_dice_list = list()
                base1_cup_dice_list = list()
                # base1
                with torch.no_grad():
                    model.eval()
                    for iter, batch in enumerate(base1_dataloader):
                        name = batch['name']
                        base1_data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                        base1_seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
                        # torchvision.utils.save_image(base1_data[0], str(iter)+'_1.jpg', padding=0)
                        # torchvision.utils.save_image(base1_seg[0], str(iter)+'_1.png', padding=0)
                        with autocast():
                            base1_output = model(base1_data, tau=tau)
                            base1_loss = criterion(base1_output[:, 0], (base1_seg[:, 0] > 0) * 1.0) + criterion(
                                base1_output[:, 1], (base1_seg[:, 0] == 2) * 1.0)
                        base1_loss_list.append(base1_loss.detach().cpu().numpy())
                        base1_output_sigmoid = torch.sigmoid(base1_output)
                        base1_disc_dice_list.append(
                            get_hard_dice(base1_output_sigmoid[:, 0].cpu(), (base1_seg[:, 0] > 0).cpu() * 1.0))
                        base1_cup_dice_list.append(
                            get_hard_dice(base1_output_sigmoid[:, 1].cpu(), (base1_seg[:, 0] == 2).cpu() * 1.0))
                            
                        
                        name_list = name[0].split('/')
                        output_oc = torch.sigmoid(base1_output[0,0])
                        output_od = torch.sigmoid(base1_output[0,1])
                        zero = torch.zeros_like(output_oc)
                        one = torch.ones_like(output_oc)
                        output_oc = torch.where(output_oc > 0.5, one, zero)
                        output_od = torch.where(output_od > 0.5, one, zero)
                            
                        
                        mask = base1_seg[0,0].cpu().numpy()
                        segg = np.uint8((mask > 0) * 255)
                        image = Image.fromarray(segg, 'L')
                        image.save('save_img/'+name_list[-3]+'/'+name_list[-2]+'/' + name_list[-1]+'_oc.png')
                        
                        mask = base1_seg[0,0].cpu().numpy()
                        segg = np.uint8((mask == 2) * 255)
                        image = Image.fromarray(segg, 'L')
                        image.save('save_img/'+name_list[-3]+'/'+name_list[-2]+'/' + name_list[-1]+'_od.png')
            
            
                        mask = output_oc.cpu().detach().numpy()
                        segg = np.uint8(mask * 255)          
                        image = Image.fromarray(segg, 'L')
                        image.save('save_img/'+name_list[-3]+'/'+name_list[-2]+'/' + name_list[-1]+'_occ.png')
                        
                        mask = output_od.cpu().detach().numpy()
                        segg = np.uint8(mask * 255)          
                        image = Image.fromarray(segg, 'L')
                        image.save('save_img/'+name_list[-3]+'/'+name_list[-2]+'/' + name_list[-1]+'_odd.png')
                        
                        

                mean_base1_loss = np.mean(base1_loss_list)
                mean_base1_disc_dice = np.mean(base1_disc_dice_list)
                mean_base1_cup_dice = np.mean(base1_cup_dice_list)
                print('  Base loss: {}\n'
                      '  Base1 disc dice: {}; Cup dice: {}'.format(mean_base1_loss, mean_base1_disc_dice,
                                                                   mean_base1_cup_dice))

                # base2
                base2_loss_list = list()
                base2_disc_dice_list = list()
                base2_cup_dice_list = list()
                with torch.no_grad():
                    model.eval()
                    for iter, batch in enumerate(base2_dataloader):
                        name = batch['name']
                        base2_data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                        base2_seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
                        with autocast():
                            base2_output = model(base2_data, tau=tau)
                            base2_loss = criterion(base2_output[:, 0], (base2_seg[:, 0] > 0) * 1.0) + criterion(
                                base2_output[:, 1], (base2_seg[:, 0] == 2) * 1.0)
                        base2_loss_list.append(base2_loss.detach().cpu().numpy())
                        base2_output_sigmoid = torch.sigmoid(base2_output)
                        base2_disc_dice_list.append(
                            get_hard_dice(base2_output_sigmoid[:, 0].cpu(), (base2_seg[:, 0] > 0).cpu() * 1.0))
                        base2_cup_dice_list.append(
                            get_hard_dice(base2_output_sigmoid[:, 1].cpu(), (base2_seg[:, 0] == 2).cpu() * 1.0))
                            
                            
                        
                        name_list = name[0].split('/')    
                        output_oc = torch.sigmoid(base2_output[0,0])
                        output_od = torch.sigmoid(base2_output[0,1])
                        zero = torch.zeros_like(output_oc)
                        one = torch.ones_like(output_oc)
                        output_oc = torch.where(output_oc > 0.5, one, zero)
                        output_od = torch.where(output_od > 0.5, one, zero)
                            
                        
                        mask = base2_seg[0,0].cpu().numpy()
                        segg = np.uint8((mask > 0) * 255)
                        image = Image.fromarray(segg, 'L')
                        image.save('save_img/'+name_list[-3]+'/'+name_list[-2]+'/' + name_list[-1]+'_oc.png')
                        
                        mask = base2_seg[0,0].cpu().numpy()
                        segg = np.uint8((mask == 2) * 255)
                        image = Image.fromarray(segg, 'L')
                        image.save('save_img/'+name_list[-3]+'/'+name_list[-2]+'/' + name_list[-1]+'_od.png')
            
            
                        mask = output_oc.cpu().detach().numpy()
                        segg = np.uint8(mask * 255)          
                        image = Image.fromarray(segg, 'L')
                        image.save('save_img/'+name_list[-3]+'/'+name_list[-2]+'/' + name_list[-1]+'_occ.png')
                        
                        mask = output_od.cpu().detach().numpy()
                        segg = np.uint8(mask * 255)          
                        image = Image.fromarray(segg, 'L')
                        image.save('save_img/'+name_list[-3]+'/'+name_list[-2]+'/' + name_list[-1]+'_odd.png')
                            

                mean_base2_loss = np.mean(base2_loss_list)
                mean_base2_disc_dice = np.mean(base2_disc_dice_list)
                mean_base2_cup_dice = np.mean(base2_cup_dice_list)
                print('  Base2 loss: {}\n'
                      '  Base2 disc dice: {}; Cup dice: {}'.format(mean_base2_loss, mean_base2_disc_dice,
                                                                   mean_base2_cup_dice))
                base3_loss_list = list()
                base3_disc_dice_list = list()
                base3_cup_dice_list = list()
                # base3
                with torch.no_grad():
                    model.eval()
                    for iter, batch in enumerate(base3_dataloader):
                        name = batch['name']
                        base3_data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                        base3_seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
                        # torchvision.utils.save_image(base3_data[0], str(iter)+'_3.jpg', padding=0)
                        # torchvision.utils.save_image(base3_seg[0], str(iter)+'_3.png', padding=0)
                        with autocast():
                            base3_output = model(base3_data, tau=tau)
                            base3_loss = criterion(base3_output[:, 0], (base3_seg[:, 0] > 0) * 1.0) + criterion(
                                base3_output[:, 1], (base3_seg[:, 0] == 2) * 1.0)
                        base3_loss_list.append(base3_loss.detach().cpu().numpy())
                        base3_output_sigmoid = torch.sigmoid(base3_output)
                        base3_disc_dice_list.append(
                            get_hard_dice(base3_output_sigmoid[:, 0].cpu(), (base3_seg[:, 0] > 0).cpu() * 1.0))
                        base3_cup_dice_list.append(
                            get_hard_dice(base3_output_sigmoid[:, 1].cpu(), (base3_seg[:, 0] == 2).cpu() * 1.0))
                            
                        
                        name_list = name[0].split('/')    
                        output_oc = torch.sigmoid(base3_output[0,0])
                        output_od = torch.sigmoid(base3_output[0,1])
                        zero = torch.zeros_like(output_oc)
                        one = torch.ones_like(output_oc)
                        output_oc = torch.where(output_oc > 0.5, one, zero)
                        output_od = torch.where(output_od > 0.5, one, zero)
                            
                        
                        mask = base3_seg[0,0].cpu().numpy()
                        segg = np.uint8((mask > 0) * 255)
                        image = Image.fromarray(segg, 'L')
                        image.save('save_img/'+name_list[-3]+'/'+name_list[-2]+'/' + name_list[-1]+'_oc.png')
                        
                        mask = base3_seg[0,0].cpu().numpy()
                        segg = np.uint8((mask == 2) * 255)
                        image = Image.fromarray(segg, 'L')
                        image.save('save_img/'+name_list[-3]+'/'+name_list[-2]+'/' + name_list[-1]+'_od.png')
            
            
                        mask = output_oc.cpu().detach().numpy()
                        segg = np.uint8(mask * 255)          
                        image = Image.fromarray(segg, 'L')
                        image.save('save_img/'+name_list[-3]+'/'+name_list[-2]+'/' + name_list[-1]+'_occ.png')
                        
                        mask = output_od.cpu().detach().numpy()
                        segg = np.uint8(mask * 255)          
                        image = Image.fromarray(segg, 'L')
                        image.save('save_img/'+name_list[-3]+'/'+name_list[-2]+'/' + name_list[-1]+'_odd.png')
                            
                            
                mean_base3_loss = np.mean(base3_loss_list)
                mean_base3_disc_dice = np.mean(base3_disc_dice_list)
                mean_base3_cup_dice = np.mean(base3_cup_dice_list)
                print('  Base3 loss: {}\n'
                      '  Base3 disc dice: {}; Cup dice: {}'.format(mean_base3_loss, mean_base3_disc_dice,
                                                                   mean_base3_cup_dice))

                mean_disc = (mean_base1_disc_dice*len(base1_dataloader)+mean_base2_disc_dice*len(base2_dataloader)
                             +mean_base3_disc_dice*len(base3_dataloader))/(len(base1_dataloader)+len(base2_dataloader)+len(base3_dataloader))
                mean_cup = (mean_base1_cup_dice*len(base1_dataloader)+mean_base2_cup_dice*len(base2_dataloader)
                            +mean_base3_cup_dice*len(base3_dataloader))/(len(base1_dataloader)+len(base2_dataloader)+len(base3_dataloader))
                print(' All dosc dice: {}; cup dice: {}'.format(mean_disc,mean_cup))

        if epoch % save_interval == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format('latest'))
            torch.save(saved_model, join(model_folder, 'model_latest.model'))

        if (epoch+1) % 100 == 0:
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
    
    
    
    
    torch.save(model, 'layer2_BinRushed1.pth')
    
    
    

    # inference
    #from inference.inference_nets.inference_unet_ccsdg import inference
    #for ts_csv_path in ts_csv:
        #inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
        #print("Running inference: {}".format(inference_tag))
        #inference('model_final.model', gpu, log_folder, patch_size, root_folder, [ts_csv_path], inference_tag, tau=tau)

