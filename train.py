import torch
import torchvision
from torchvision import transforms

from model import RAE
from network import *
from autoencoder_helpers import makedirs

from data_preprocessing import batch_elastic_transform

import time

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import datetime
from torch.utils.tensorboard import SummaryWriter

from rtpt.rtpt import RTPT

import torchvision.transforms.functional as TF

from PIL import Image

config = dict({
    'device': 'cuda:14',
    
    'save_step': 100,
    'print_step': 100,
    'display_step': 1,

    'lambda_ae_proto': 10,
    'lambda_ae': 0,
    'lambda_r1': 1,
    'lambda_r2': 1,
    'lambda_r3': 1,
    'lambda_r4': 1,
    'lambda_r5': 0, # 1e-2,

    'learning_rate': 1e-3,
    'training_epochs': 5000,
    'batch_size': 500,
    'n_workers': 4,

    'n_prototype_vectors': 10,

    'batch_elastic_transform': False,
    'sigma': 4,
    'alpha': 20,

    'img_shape': (28,28),

    'experiment_name': '',
    'results_dir': 'results',
    'model_dir': 'states',
    'img_dir': 'imgs'
})

if config['experiment_name'] == '':
    config['experiment_name'] = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').split('.')[0]

config['results_dir'] = os.path.join(config['results_dir'], config['experiment_name'])
config['model_dir'] = os.path.join(config['results_dir'],  config['model_dir'])
config['img_dir'] = os.path.join(config['results_dir'], config['img_dir'])
makedirs(config['model_dir'])
makedirs(config['img_dir'])

import json
with open(os.path.join(config['results_dir'], 'args.txt'), 'w') as json_file:
    json.dump(config, json_file, indent=2)


writer = SummaryWriter(log_dir=config['results_dir'])  

# dataloader setup
mnist_data = torchvision.datasets.MNIST(root='dataset', train=True, download=True, transform=transforms.ToTensor())
mnist_data_test = torchvision.datasets.MNIST(root='dataset', train=False, download=True, transform=transforms.ToTensor())

mnist_data_concat = torch.utils.data.ConcatDataset((mnist_data, mnist_data_test))

data_loader = torch.utils.data.DataLoader(mnist_data_concat, batch_size=config['batch_size'], shuffle=True, num_workers=config['n_workers'], pin_memory=True)

# model setup
model = RAE(input_dim=(1,1,config['img_shape'][0], config['img_shape'][1]), n_prototype_vectors=config['n_prototype_vectors'])
model = model.to(config['device'])

# optimizer setup
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

rtpt = RTPT(name_initials='MM', experiment_name='XIC_PrototypeDL', max_iterations=config['training_epochs'])
rtpt.start()

for e in range(0, config['training_epochs']):
    max_iter = len(data_loader)
    start = time.time()
    loss_dict = dict({'ae_proto_loss':0, 'r1_loss':0, 'r2_loss':0, 'r3_loss':0, 'r4_loss':0, 'r5_loss':0, 'ae_loss':0, 'loss':0})

    for i, batch in enumerate(data_loader):
        imgs = batch[0]
        # to apply elastic transform, batch has to be flattened
        # store original size to put it back into orginial shape after transformation
        imgs_shape = imgs.shape
        if config['batch_elastic_transform']:
            imgs = batch_elastic_transform(imgs.view(config['batch_size'], -1), sigma=config['sigma'], alpha=config['alpha'], height=config['img_shape'][0], width=config['img_shape'][1])
            imgs = torch.reshape(torch.tensor(imgs), imgs_shape)
        imgs = imgs.to(config['device'])

        labels = batch[1].to(config['device'])

        optimizer.zero_grad()        

        pred = model.forward(imgs)

        prototype_vectors = model.prototype_layer.prototype_vectors
        feature_vectors = model.feature_vectors

        # draws prototype close to training example
        r1_loss = torch.mean(torch.min(list_of_distances(prototype_vectors, feature_vectors.view(-1, model.input_dim_prototype)), dim=1)[0])
        # draws encoding close to prototype
        r2_loss = torch.mean(torch.min(list_of_distances(feature_vectors.view(-1, model.input_dim_prototype ), prototype_vectors), dim=1)[0])
        
        # draws encoding before the prototype layer close to encoding after the prototype layer
        r3_loss = torch.mean(torch.min(list_of_distances(feature_vectors.view(-1, model.input_dim_prototype), pred.view(-1, model.input_dim_prototype)), dim=1)[0])
        # draws encoding after the prototype layer close to encoding before the prototype layer
        r4_loss = torch.mean(torch.min(list_of_distances(pred.view(-1, model.input_dim_prototype), feature_vectors.view(-1, model.input_dim_prototype )), dim=1)[0])

        # draw prototypes away from each other
        # get absolute values of lower triangle without diagonal (distances between prototype_vectors)
        # diagonal would be distance to itself, distance matrix is symmetric
        # r5_loss = -torch.mean(torch.tril(list_of_distances(prototype_vectors, prototype_vectors), diagonal=1))
        r5_loss = -torch.mean(list_of_distances(prototype_vectors, prototype_vectors))

        rec = model.forward_dec(feature_vectors)
        ae_loss = torch.mean(list_of_norms(rec-imgs))

        rec_proto = model.forward_dec(pred)
        ae_proto_loss = torch.mean(list_of_norms(rec-imgs))

        loss = config['lambda_ae_proto'] * ae_proto_loss +\
                config['lambda_r1'] * r1_loss +\
                config['lambda_r2'] * r2_loss +\
                config['lambda_r3'] * r3_loss +\
                config['lambda_r4'] * r4_loss +\
                config['lambda_r5'] * r5_loss +\
                config['lambda_ae'] * ae_loss

        loss.backward()
        
        optimizer.step()

        loss_dict['ae_proto_loss'] += ae_proto_loss.item()
        loss_dict['r1_loss'] += r1_loss.item()
        loss_dict['r2_loss'] += r2_loss.item()
        loss_dict['r3_loss'] += r3_loss.item()
        loss_dict['r4_loss'] += r4_loss.item()
        loss_dict['r5_loss'] += r5_loss.item()
        loss_dict['ae_loss'] += ae_loss.item()
        loss_dict['loss'] += loss.item()

    for key in loss_dict.keys():
        loss_dict[key] /= len(data_loader)

    rtpt.step(subtitle=f'loss={loss_dict["loss"]:2.2f}')

    if (e+1) % config['display_step'] == 0 or e == config['training_epochs'] - 1:
        for key in loss_dict.keys():
            writer.add_scalar(f'train/{key}', loss_dict[key], global_step=e)

    if (e+1) % config['print_step'] == 0 or e == config['training_epochs'] - 1:
        print(f'epoch {e} - loss {loss.item():2.4f} - time/epoch {(time.time()-start):2.2f}')
        loss_summary = ''
        for key in loss_dict.keys():
            loss_summary += f'{key} {loss_dict[key]:2.2f} '
        # print(f'ae_proto_loss {round(ae_proto_loss.item(),3)} ae_loss {round(ae_loss.item(),3)} r1 {round(r1_loss.item(),3)} r2 {round(r2_loss.item(),3)} r3 {round(r3_loss.item(),3)} r4 {round(r4_loss.item(),3)} r5 {round(r5_loss.item(),3)}')
        print(loss_summary)
        
    if (e+1) % config['save_step'] == 0 or e == config['training_epochs'] - 1:

        state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ep': e
                }
        torch.save(state, os.path.join(config['model_dir'], '%05d.pth' % (e)))

        # decode prototype vectors
        prototype_imgs = model.forward_dec(prototype_vectors.reshape((-1,10,2,2))).detach().cpu()

        # visualize the prototype images
        n_cols = 5
        n_rows = config['n_prototype_vectors'] // n_cols + 1 if config['n_prototype_vectors'] % n_cols != 0 else config['n_prototype_vectors'] // n_cols
        g, b = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
        for i in range(n_rows):
            for j in range(n_cols):
                if i*n_cols + j < config['n_prototype_vectors']:
                    b[i][j].imshow(prototype_imgs[i*n_cols + j].reshape(config['img_shape'][0], config['img_shape'][1]),
                                    cmap='gray',
                                    interpolation='none')
                    b[i][j].axis('off')
                    
        img_save_path = os.path.join(config['img_dir'], f'{e:05d}' + '_prototype_result' + '.png')
        plt.savefig(img_save_path,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        image = Image.open(img_save_path)
        image = TF.to_tensor(image)
        writer.add_image(f'train/prototype_result', image, global_step=e)

        # apply encoding and decoding over a small subset of the training set
        imgs = []
        for batch in data_loader:
            imgs = batch[0].to(config['device'])
            break

        examples_to_show = 10
        
        encoded = model.enc.forward(imgs[:examples_to_show])
        decoded = model.dec.forward(encoded)

        encoded_proto = model.forward(imgs[:examples_to_show])
        decoded_proto = model.dec.forward(encoded_proto)
    
        decoded = decoded.detach().cpu()
        decoded_proto = decoded_proto.detach().cpu()
        imgs = imgs.detach().cpu()

        # compare original images to their reconstructions
        f, a = plt.subplots(3, examples_to_show, figsize=(examples_to_show, 3))
        for i in range(examples_to_show):
            a[0][i].imshow(imgs[i].reshape(config['img_shape'][0], config['img_shape'][1]),
                            cmap='gray',
                            interpolation='none')
            a[0][i].axis('off')

            a[1][i].imshow(decoded[i].reshape(config['img_shape'][0], config['img_shape'][1]), 
                            cmap='gray',
                            interpolation='none')
            a[1][i].axis('off')
            
            a[2][i].imshow(decoded_proto[i].reshape(config['img_shape'][0], config['img_shape'][1]), 
                            cmap='gray',
                            interpolation='none')
            a[2][i].axis('off')

        img_save_path = os.path.join(config['img_dir'], f'{e:05d}' + '_decoding_result' + '.png')
        plt.savefig(img_save_path,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        image = Image.open(img_save_path)
        image = TF.to_tensor(image)
        writer.add_image(f'train/decoding_result', image, global_step=e)

        print(f'SAVED - epoch {e} - imgs @ {config["img_dir"]} - model @ {config["model_dir"]}')
