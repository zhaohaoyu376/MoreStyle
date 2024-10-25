from time import time
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data
import torch.nn as nn
from utils.file_utils import check_folders
from batchgenerators.utilities.file_and_folder_operations import *
# from models.unet_ccsdg import UNetCCSDG, Projector
from models.unet_mixnoise import UNet_Mix,generater
from models.noise_encoder import NoiseEncoder
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list
from datasets.utils.transform import collate_fn_tr_styleaug, collate_fn_ts
from utils.lr import adjust_learning_rate,adjust_weight
from utils.metrics.dice import get_hard_dice
from torchvision.utils import make_grid
from utils.loss import fourier_exchage_loss
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils import clip_grad_norm_

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


def train(args):
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

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu])
    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    writer = SummaryWriter(log_dir=tensorboard_folder)

    source_name = args.Source_Dataset
    source_csv = []
    test_csv = []
    for s_n in source_name:
        source_csv.append(s_n + '_train.csv')
        test_csv.append(s_n + '_train.csv')

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
                                                drop_last=True,
                                                collate_fn=collate_fn_tr_styleaug)
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads,
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

    model = UNet_Mix()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    Generater = generater(mixstyle_layers=args.mixstyle_layers)
    Generater.to(device)

    noise_encoder = NoiseEncoder(input_channel=3)
    noise_encoder.to(device)


    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.99, nesterov=True)
    optimizer_generater = torch.optim.SGD(Generater.parameters(), lr=initial_lr, momentum=0.99, nesterov=True)
    optimizer_noise = torch.optim.SGD(list(Generater.noiseEncoder.parameters()), lr=1e-2, momentum=0.99, nesterov=True)

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

        train_loss_list0 = list()
        train_loss_list1 = list()
        train_loss_list2 = list()
        train_loss_list3 = list()
        train_loss_list4 = list()
        train_loss_list5 = list()
        train_loss_list6 = list()
        train_disc_dice_list = list()
        train_cup_dice_list = list()
        content_loss_list = list()
        style_loss_list = list()
        for iter, batch in enumerate(tr_dataloader):
            name = batch['name']
            data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
            fda_data = torch.from_numpy(batch['fda_data']).cuda().to(dtype=torch.float32)
            GLA_data = torch.from_numpy(batch['GLA']).cuda().to(dtype=torch.float32)
            seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)

            # optimizer_prompt.zero_grad()
            # with autocast():
            #     f_content, f_style = model.forward_first_layer(data, tau=tau)
            #     f_content_fda, f_style_fda = model.forward_first_layer(fda_data, tau=tau)
            #     f_content_GLA, f_style_GLA = model.forward_first_layer(GLA_data, tau=tau)
            #     f_content_p = projector(f_content)
            #     f_style_p = projector(f_style)
            #     f_content_fda_p = projector(f_content_fda)
            #     f_style_fda_p = projector(f_style_fda)
            #     f_content_GLA_p = projector(f_content_GLA)
            #     f_style_GLA_p = projector(f_style_GLA)
            #     content_loss = F.l1_loss(f_content_p, f_content_fda_p, reduction='mean') + \
            #                    F.l1_loss(f_content_fda_p, f_content_p, reduction='mean') + \
            #                    F.l1_loss(f_content_p, f_content_GLA_p, reduction='mean') + \
            #                    F.l1_loss(f_content_GLA_p, f_content_p, reduction='mean') + \
            #                    F.l1_loss(f_content_GLA_p, f_content_fda_p, reduction='mean') + \
            #                    F.l1_loss(f_content_fda_p, f_content_GLA_p, reduction='mean')
            #     style_loss = F.l1_loss(f_style_p, f_style_fda_p, reduction='mean') + \
            #                  F.l1_loss(f_style_fda_p, f_style_p, reduction='mean') + \
            #                  F.l1_loss(f_style_p, f_style_GLA_p, reduction='mean') + \
            #                  F.l1_loss(f_style_GLA_p, f_style_p, reduction='mean') + \
            #                  F.l1_loss(f_style_GLA_p, f_style_fda_p, reduction='mean') + \
            #                  F.l1_loss(f_style_fda_p, f_style_GLA_p, reduction='mean')
            #     style_loss = - style_loss
            # amp_grad_scaler.scale(content_loss + style_loss).backward()
            # amp_grad_scaler.unscale_(optimizer_prompt)
            # amp_grad_scaler.step(optimizer_prompt)
            # amp_grad_scaler.update()


            optimizer.zero_grad()
            with autocast():
                output,_ = model(data)
                seg_loss = criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + \
                       criterion(output[:, 1], (seg[:, 0] == 2) * 1.0)
            amp_grad_scaler.scale(seg_loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
            train_loss_list0.append(seg_loss.detach().cpu().numpy())

            # optimizer_noise.zero_grad()
            # with autocast():
            #     randon = torch.randn_like(data)
            #     noise = noise_encoder(randon)
            #     x, sfs = model.encoder(data)
            #     recon_img = Generater(x, sfs, noise)
            #     # output, _ = model(recon_img.data)
            #     # output = output.float()
            #     seg = seg.float()
            #     seg_loss_n = -criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) - \
            #                criterion(output[:, 1], (seg[:, 0] == 2) * 1.0)
            # amp_grad_scaler.scale(seg_loss_n).backward()
            # amp_grad_scaler.unscale_(optimizer_noise)
            # amp_grad_scaler.step(optimizer_noise)
            # amp_grad_scaler.update()
            # train_loss_list1.append(seg_loss_n.detach().cpu().numpy())

            optimizer_noise.zero_grad()
            x, sfs = model.encoder(data)
            recon_img = Generater(x, sfs, data)
            l_l = adjust_weight(epoch,1e-2)
            _, recon_img = fourier_exchage_loss(recon_img, data)
            output, _ = model(recon_img.data)
            seg_loss_n = -criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) - \
                           criterion(output[:, 1], (seg[:, 0] == 2) * 1.0)
            seg_loss_n.backward()
            optimizer_noise.step()
            train_loss_list1.append(seg_loss_n.detach().cpu().numpy())

            # optimizer_generater.zero_grad()
            # x, sfs = model.encoder(data)
            # output = Generater(x,sfs,data)
            # Fourier_loss,recon_img = fourier_exchage_loss(output.data,data)
            # mse_loss = nn.MSELoss()(output.data,data)
            # recon_loss = 0.1 * mse_loss + Fourier_loss
            # # recon_loss = Fourier_loss
            # recon_loss.requires_grad_(True)
            # recon_loss.backward()
            # optimizer_generater.step()
            # train_loss_list2.append(Fourier_loss.detach().cpu().numpy())
            # train_loss_list3.append(mse_loss.detach().cpu().numpy())

            optimizer_generater.zero_grad()
            with autocast():
                x, sfs = model.encoder(data)
                output = Generater(x,sfs,data)
                Fourier_loss, recon_img = fourier_exchage_loss(output, data)
                # Fourier_loss = fourier_loss(output,data)
                # output = torch.tensor(output, dtype=torch.int)
                # recon_img = torch.tensor(recon_img, dtype=torch.int)
                mse_loss = nn.MSELoss()(output,data)
                gradient_output = gradient(output)
                gradient_data = gradient(data)
                gradient_loss = nn.MSELoss()(gradient_output,gradient_data)
                weight = adjust_weight(epoch)
                recon_loss = weight*mse_loss+Fourier_loss+weight*gradient_loss
                # recon_loss = Fourier_loss
            amp_grad_scaler.scale(recon_loss).backward()
            amp_grad_scaler.unscale_(optimizer_generater)
            amp_grad_scaler.step(optimizer_generater)
            amp_grad_scaler.update()
            train_loss_list2.append(Fourier_loss.detach().cpu().numpy())
            train_loss_list3.append(mse_loss.detach().cpu().numpy())

            # optimizer_generater.zero_grad()
            # with autocast():
            #     x, sfs = model.encoder(data)
            #     output = Generater(x,sfs,noise)
            #     Fourier_loss = fourier_loss(output.data,data)
            #     mse_loss = nn.MSELoss()(output.data,data)
            #     recon_loss = 0.01 * mse_loss + Fourier_loss
            #     recon_loss.requires_grad_(True)
            # amp_grad_scaler.scale(recon_loss).backward()
            # amp_grad_scaler.unscale_(optimizer_generater)
            # amp_grad_scaler.step(optimizer_generater)
            # amp_grad_scaler.update()
            # train_loss_list1.append(Fourier_loss.detach().cpu().numpy())
            # train_loss_list2.append(mse_loss.detach().cpu().numpy())

            rec_img = output.detach().clone()
            name_list = name[0].split('/')


            torchvision.utils.save_image(data[0], name_list[-1] + '.png', padding=0)
            torchvision.utils.save_image(rec_img[0], name_list[-1] + '_rec.png', padding=0)
            torchvision.utils.save_image(recon_img[0], name_list[-1] + '_exchange.png', padding=0)
            

            optimizer.zero_grad()
            with autocast():
                output,_ = model(recon_img.data)
                seg_loss_f = criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + \
                       criterion(output[:, 1], (seg[:, 0] == 2) * 1.0)
            amp_grad_scaler.scale(seg_loss_f).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
            train_loss_list4.append(seg_loss_f.detach().cpu().numpy())
            
            optimizer.zero_grad()
            with autocast():
                output,_ = model(rec_img.data)
                seg_loss_f = criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + \
                       criterion(output[:, 1], (seg[:, 0] == 2) * 1.0)
            amp_grad_scaler.scale(seg_loss_f).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()


            # optimizer.zero_grad()
            # with autocast():
            #     _,fda_content = model(fda_data)
            #     _,gla_content = model(GLA_data)
            #     _,content = model(data)
            #     loss = F.l1_loss(content, fda_content, reduction='mean') + \
            #            F.l1_loss(content, gla_content, reduction='mean') + \
            #            F.l1_loss(gla_content, fda_content, reduction='mean')
            # amp_grad_scaler.scale(loss).backward()
            # amp_grad_scaler.unscale_(optimizer)
            # amp_grad_scaler.step(optimizer)
            # amp_grad_scaler.update()

            output_sigmoid = torch.sigmoid(output)
            train_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
            train_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
            del seg

        mean_tr_loss0 = np.mean(train_loss_list0)
        mean_tr_loss1 = np.mean(train_loss_list1)
        mean_tr_loss2 = np.mean(train_loss_list2)
        mean_tr_loss3 = np.mean(train_loss_list3)
        mean_tr_loss4 = np.mean(train_loss_list4)
        mean_tr_loss5 = np.mean(train_loss_list5)
        mean_tr_loss6 = np.mean(train_loss_list6)
        mean_content_loss = np.mean(content_loss_list)
        mean_style_loss = np.mean(style_loss_list)
        mean_tr_disc_dice = np.mean(train_disc_dice_list)
        mean_tr_cup_dice = np.mean(train_cup_dice_list)
        writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss4, epoch)
        writer.add_scalar("Train Scalars/Train Content Loss", mean_content_loss, epoch)
        writer.add_scalar("Train Scalars/Train Style Loss", mean_style_loss, epoch)
        writer.add_scalar("Train Scalars/Disc Dice", mean_tr_disc_dice, epoch)
        writer.add_scalar("Train Scalars/Cup Dice", mean_tr_cup_dice, epoch)
        print('  seg_loss: {}; noise_loss: {}, fourier_loss:{}\n'
              '  fda_loss: {}, gla_loss:{}\n'
              '  mse_loss: {}; seg_loss_f: {}\n'
              '  Tr disc dice: {}; Cup dice: {}'.format(mean_tr_loss0,mean_tr_loss1,mean_tr_loss2,mean_tr_loss5,mean_tr_loss6,mean_tr_loss3,mean_tr_loss4, mean_tr_disc_dice, mean_tr_cup_dice))


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
                        output,_ = model(data)
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
                        base1_data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                        # torchvision.utils.save_image(torch.tensor(base1_data, str(iter)+'.jpg', padding=0))
                        # torchvision.utils.save_image(torch.tensor(base1_data, str(iter)+'.png', padding=0))
                        base1_seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
                        with autocast():
                            base1_output,_ = model(base1_data)
                            base1_loss = criterion(base1_output[:, 0], (base1_seg[:, 0] > 0) * 1.0) + criterion(
                                base1_output[:, 1], (base1_seg[:, 0] == 2) * 1.0)
                        base1_loss_list.append(base1_loss.detach().cpu().numpy())
                        base1_output_sigmoid = torch.sigmoid(base1_output)
                        base1_disc_dice_list.append(
                            get_hard_dice(base1_output_sigmoid[:, 0].cpu(), (base1_seg[:, 0] > 0).cpu() * 1.0))
                        base1_cup_dice_list.append(
                            get_hard_dice(base1_output_sigmoid[:, 1].cpu(), (base1_seg[:, 0] == 2).cpu() * 1.0))

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
                        base2_data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                        base2_seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
                        with autocast():
                            base2_output,_ = model(base2_data)
                            base2_loss = criterion(base2_output[:, 0], (base2_seg[:, 0] > 0) * 1.0) + criterion(
                                base2_output[:, 1], (base2_seg[:, 0] == 2) * 1.0)
                        base2_loss_list.append(base2_loss.detach().cpu().numpy())
                        base2_output_sigmoid = torch.sigmoid(base2_output)
                        base2_disc_dice_list.append(
                            get_hard_dice(base2_output_sigmoid[:, 0].cpu(), (base2_seg[:, 0] > 0).cpu() * 1.0))
                        base2_cup_dice_list.append(
                            get_hard_dice(base2_output_sigmoid[:, 1].cpu(), (base2_seg[:, 0] == 2).cpu() * 1.0))

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
                        base3_data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                        base3_seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
                        with autocast():
                            base3_output,_ = model(base3_data)
                            base3_loss = criterion(base3_output[:, 0], (base3_seg[:, 0] > 0) * 1.0) + criterion(
                                base3_output[:, 1], (base3_seg[:, 0] == 2) * 1.0)
                        base3_loss_list.append(base3_loss.detach().cpu().numpy())
                        base3_output_sigmoid = torch.sigmoid(base3_output)
                        base3_disc_dice_list.append(
                            get_hard_dice(base3_output_sigmoid[:, 0].cpu(), (base3_seg[:, 0] > 0).cpu() * 1.0))
                        base3_cup_dice_list.append(
                            get_hard_dice(base3_output_sigmoid[:, 1].cpu(), (base3_seg[:, 0] == 2).cpu() * 1.0))
                mean_base3_loss = np.mean(base3_loss_list)
                mean_base3_disc_dice = np.mean(base3_disc_dice_list)
                mean_base3_cup_dice = np.mean(base3_cup_dice_list)
                print('  Base3 loss: {}\n'
                      '  Base3 disc dice: {}; Cup dice: {}'.format(mean_base3_loss, mean_base3_disc_dice,
                                                                   mean_base3_cup_dice))

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

