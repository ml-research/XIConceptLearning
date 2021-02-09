import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from model.slot_attention_prototype import CAE, SlotCAE
import data.data as data
from data.data_preprocessing import batch_elastic_transform
from utils.utils import *
from utils.autoencoder_helpers import *
from utils.hessian_penalty_pytorch import hessian_penalty

from args import get_args
from rtpt.rtpt import RTPT

from multiprocessing import Pool

def train(args):
    # create directories
    results_dir = os.path.join(args.results_dir, args.name)
    args.tfb_dir = os.path.join(results_dir, args.name)
    makedirs(args.tfb_dir)
    args.model_dir=os.path.join(results_dir, 'states')
    makedirs(args.model_dir)
    args.img_dir = os.path.join(results_dir, 'imgs')
    makedirs(args.img_dir)

    writer = create_writer(args)

    # create datasets and dataloader
    clevr_data = data.CLEVR(args.data_dir, 'train', resize=(args.img_dim_h, args.img_dim_w))
    data_loader = data.get_loader(
            clevr_data,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=True,
        )

    clevr_data_test = data.CLEVR(args.data_dir, 'test', resize=(args.img_dim_h, args.img_dim_w))
    data_loader_test = data.get_loader(
            clevr_data_test,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=True,
        )
    
    # use test and train dataset
    clevr_data_concat = torch.utils.data.ConcatDataset([clevr_data, clevr_data_test])
    data_loader_concat = data.get_loader(
            clevr_data_concat,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=True,
        )
    dataloader = data_loader_concat
    
    head_config = []
    for class_select in args.classes_select:
        head_config += [dict({
            'class': class_select,
            'len': len(list(args.class_dict[class_select])),
            'idcs': list(args.class_dict[class_select])})]

    # model setup
    model = SlotCAE(input_dim=(1,args.img_dim_c, args.img_dim_h, args.img_dim_w), 
                    n_prototype_vectors=args.n_prototype_vectors, n_classes=args.n_classes, n_slots=args.n_slots,
                    head_config=head_config).to(args.device)
    model.load_pretrained_slot_attention('logs/slot-attention-clevr-objdiscovery-14')
    
    if args.freeze:
        print('Freezing slot attention object discovery encoder + decoder')
        model.freeze_slot_attention()

    # model.to(args.device)
    # model = nn.DataParallel(model, device_ids=args.device_ids)
    print('Device IDS', args.device_ids)

    model.to(args.device)
    # if not args.freeze:
    model.slot_att = nn.DataParallel(model.slot_att, device_ids=args.device_ids)
    model.class_heads = nn.ModuleList([nn.DataParallel(class_head, device_ids=args.device_ids) for class_head in model.class_heads])

    # optimizer setup
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if not args.freeze:
        optimizer_slot_att = torch.optim.Adam(model.slot_att.parameters(), lr=args.lr*0.1)
    optimizer_class_heads = torch.optim.Adam(model.class_heads.parameters(), lr=args.lr)

    # scheduler = get_scheduler(optimizer, args, cur_ep=-1)
    if not args.freeze:
        scheduler_slot_att = get_scheduler(optimizer_slot_att, args, cur_ep=-1)
    scheduler_class_heads = get_scheduler(optimizer_class_heads, args, cur_ep=-1)

    # create RTPT object
    rtpt = RTPT(name_initials='MM', experiment_name='slot_att_prototype', max_iterations=args.epochs)
    rtpt.start()

    for e in range(0, args.epochs):

        max_iter = len(data_loader)
        start = time.time()

        hung_loss, r1_loss, r2_loss, recon_loss, w_loss, avg_prec = 0, 0, 0, 0, 0, 0
        hung_loss_head, r1_loss_head, r2_loss_head = np.zeros(len(model.class_heads)), np.zeros(len(model.class_heads)), np.zeros(len(model.class_heads))# [], [] 

        for i, batch in enumerate(data_loader):
        
            imgs = batch[0]

            # to apply elastic transform, batch has to be flattened
            # store original size to put it back into orginial shape after transformation
            if args.batch_elastic_transform:
                imgs_shape = imgs.shape
                sigma = 4
                alpha = 20
                imgs = batch_elastic_transform(imgs.view(args.batch_size, -1), sigma=sigma, alpha=alpha, height=args.img_dim_h, width=args.img_dim_w)
                imgs = torch.reshape(torch.tensor(imgs), imgs_shape)
            imgs = imgs.to(args.device)

            labels = batch[1].to(args.device) 

            pred = model.forward(imgs)

            # hungarian loss per head
            hung_loss_head_batch = []
            hung_loss_batch = torch.zeros(1).to(args.device)
            for config, pred in zip(head_config, pred):
                # print(pred.detach().cpu().numpy().shape, labels[:,:,config['idcs']].expand(pred.shape).shape)
                # avg_prec = average_precision_clevr(pred.detach().cpu().numpy(), labels[:,:,config['idcs']].expand(pred.shape).detach().cpu().numpy(), -1)
                # print('AVG P', avg_prec)
                with Pool(args.n_slots) as p:
                    loss_tmp = hungarian_loss(pred, labels[:,:,config['idcs']], p)
                    hung_loss_head_batch += [loss_tmp]
                    hung_loss_batch += loss_tmp
            
            r1_loss_head_batch = []
            r1_loss_batch = torch.zeros(1).to(args.device)
            r2_loss_head_batch = []
            r2_loss_batch = torch.zeros(1).to(args.device)
            feature_vectors = model.forward_enc(imgs)

            for i, head in enumerate(model.class_heads):                
                prototype_vectors_tmp = head(head(None, op='prototype_vectors'), op='lin_trans_inv')
                feature_vectors_tmp = feature_vectors[:,:,i*model.head_input_dim:(i+1)*model.head_input_dim] 
                feature_vectors_tmp = head(feature_vectors_tmp, op='lin_trans')
                
                # r1
                loss_tmp = torch.mean(torch.min(list_of_distances(prototype_vectors_tmp, feature_vectors_tmp.reshape(-1, prototype_vectors_tmp.shape[-1])), dim=1)[0])
                r1_loss_head_batch += [loss_tmp]
                r1_loss_batch += loss_tmp
                
                # r2
                if not args.freeze:
                    loss_tmp = torch.mean(torch.min(list_of_distances(feature_vectors_tmp.reshape(-1, prototype_vectors_tmp.shape[-1]), prototype_vectors_tmp), dim=1)[0])
                    r2_loss_head_batch += [loss_tmp]
                    r2_loss_batch += loss_tmp
                else:
                    loss_tmp = torch.zeros(1).to(args.device)
                    r2_loss_head_batch += [loss_tmp]

            # reconstruction loss for slot_att_obj_discorvery
            if not args.freeze:
                criterion = torch.nn.MSELoss()
                recon_combined, recons, masks = model.forward_dec(feature_vectors)
                recon_loss_batch = criterion(imgs, recon_combined)
            else:
                recon_loss_batch = torch.zeros(1).to(args.device)

            loss = args.lambda_hung * hung_loss_batch +\
                    args.lambda_r1 * r1_loss_batch +\
                    args.lambda_r2 * r2_loss_batch +\
                    args.lambda_recon * recon_loss_batch
            
            if not args.freeze:
                optimizer_slot_att.zero_grad()
            optimizer_class_heads.zero_grad()

            loss.backward()

            if not args.freeze:
                optimizer_slot_att.step()
            optimizer_class_heads.step()

            # average precision
            avg_prec_batch = 0 # average_precision_clevr(pred.detach().cpu().numpy(), labels.detach().cpu().numpy(), -1)
            avg_prec += avg_prec_batch
            # combined losses
            hung_loss += hung_loss_batch
            r1_loss += r1_loss_batch
            r2_loss += r2_loss_batch
            recon_loss += recon_loss_batch
            w_loss += loss # weighted loss
            # head losses
            hung_loss_head += [l.item() for l in hung_loss_head_batch]
            r1_loss_head += [l.item() for l in r1_loss_head_batch]
            r2_loss_head += [l.item() for l in r2_loss_head_batch]

        # average precision
        avg_prec /= max_iter
        # combined losses
        hung_loss /= max_iter
        r1_loss /= max_iter
        r2_loss /= max_iter
        recon_loss /= max_iter
        w_loss /= max_iter
        # head losses
        hung_loss_head = [l/max_iter for l in hung_loss_head]
        r1_loss_head = [l/max_iter for l in r1_loss_head]
        r2_loss_head = [l/max_iter for l in r2_loss_head]

        # update learning rate
        if not args.freeze:
            scheduler_slot_att.step()
        scheduler_class_heads.step()

        if e < args.warm_up_steps:
            if not args.freeze:
                optimizer_slot_att.param_groups[0]["lr"] = args.lr * (e+1)/args.warm_up_steps
            optimizer_class_heads.param_groups[0]["lr"] = args.lr * (e+1)/args.warm_up_steps
        
        # RTPT step
        rtpt.step(subtitle=f"loss={loss.item():2.2f}")

        if (e+1) % args.save_step == 0 or e == args.epochs - 1:
            with torch.no_grad():            
                # save model states
                state = {
                        'model': model.state_dict(),
                        'optimizer_class_heads': optimizer_class_heads.state_dict(),
                        'ep': e
                        }
                if not args.freeze:
                    state['optimizer_slot_att'] = optimizer_slot_att.state_dict()
                
                torch.save(state, os.path.join(args.model_dir, '%05d.pth' % (e)))

                for i, head in enumerate(model.class_heads):                

                    prototype_slot_dim = head.forward(head(None, op='prototype_vectors'), op='lin_trans_inv')
                    recon_combined, recons, masks = model.forward_dec(torch.unsqueeze(prototype_slot_dim, dim=1))

                    grid_img = torchvision.utils.make_grid(recon_combined, nrow=4)
                    save_file_path = os.path.join(args.img_dir, f'ep_{e:05d}_head_{i}_proto.png')
                    torchvision.utils.save_image(grid_img, save_file_path)

                    from PIL import Image
                    image = Image.open(save_file_path)
                    trans = torchvision.transforms.ToTensor()
                    writer.add_image(f'head_{i}_proto', trans(image), global_step=e)
            
            print(f'SAVED - epoch {e} - imgs @ {args.img_dir} - model @ {args.model_dir}')

        # write to tensorboard
        if (e+1) % args.display_step == 0 or e == args.epochs - 1:
            
            writer.add_scalar('hyperparameters/learningrate', optimizer_class_heads.param_groups[0]["lr"], e)
            writer.add_scalar('loss/w_loss', w_loss.item(), e)
            writer.add_scalar('loss/hungarian', hung_loss.item(), e)
            writer.add_scalar('loss/recon_loss', recon_loss.item(), e)
            writer.add_scalar('loss/r1', r1_loss.item(), e)
            writer.add_scalar('loss/r2', r2_loss.item(), e)
            # writer.add_scalar('metric/average_precision', avg_prec, e)

            for i, (l_h, l_r1, l_r2) in enumerate(zip(hung_loss_head, r1_loss_head, r2_loss_head)):
                writer.add_scalar(f'loss/hung_loss_{i}', l_h.item(), e)
                writer.add_scalar(f'loss/r1_loss_{i}', l_r1.item(), e)
                writer.add_scalar(f'loss/r2_loss_{i}', l_r2.item(), e)

            # max_iter = len(data_loader_test)
            # start = time.time()
            # hung_loss_test = 0
            # avg_prec_test = 0
            # for i, batch in enumerate(data_loader_test):
            #     imgs_test = batch[0].to(args.device)
            #     labels_test = batch[1].to(args.device)
            #     pred_test = model.forward(imgs_test)
            #     with Pool(args.n_slots) as p:
            #         hung_loss_batch_test = hungarian_loss(pred_test, labels_test, p)
            #     hung_loss_test += hung_loss_batch_test
            #     avg_prec_batch = average_precision_clevr(pred.detach().cpu().numpy(), labels.detach().cpu().numpy(), -1)
            #     avg_prec_test += avg_prec_batch


            # hung_loss_test /= max_iter
            # avg_prec_test /= max_iter

            # writer.add_scalar('loss/hungarian_test', hung_loss_test.item(), e)
            # writer.add_scalar('metric/average_precision_test', avg_prec_test, e)

        print(f'exp {args.name} - epoch {e} - time {round(time.time()-start,2)} - loss {round(loss.item(),6)} - learning rate {optimizer_class_heads.param_groups[0]["lr"]}')

    print(f'finished training {args.name}!')

def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()