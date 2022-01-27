import torch
import time
import os
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
import torch.optim as optim
from utils import utils
from dataloader.dataloader import get_dataloader

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x

class GAN(object):
    def __init__(self, config):
        # parameters
        self.config = config
        self.input_size = config['data']['transform']['input_size']
        self.dataset = config['data']['dataloader']['dataset']
        self.batch_size = config['data']['dataloader']['batch_size']
        self.epoch = config['training']['epoch']
        self.save_dir = config['log']['save_dir']
        self.log_dir = config['log']['log_dir']
        self.z_dim = 62

        # load dataset
        self.dataloader = get_dataloader(self.config['data'])
        data_shape = self.dataloader.__iter__().__next__().shape

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data_shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data_shape[1], output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=config['training']['lrG'], betas=(config['training']['beta1'], config['training']['beta2']))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=config['training']['lrD'], betas=(config['training']['beta1'], config['training']['beta2']))
        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')


        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if torch.cuda.is_available():
            self.sample_z_ = self.sample_z_.cuda()


    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if torch.cuda.is_available():
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, x_ in enumerate(self.dataloader):
                if iter == self.dataloader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))
                if torch.cuda.is_available():
                    x_, z_ = x_.cuda(), z_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.dataloader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))
                if ((iter + 1) % 1000) == 0:
                    print('Save image...')
                    with torch.no_grad():
                        self.save_img(epoch + 1, iter + 1)
            
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        # utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
        #                          self.epoch)
        # utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def save_img(self, epoch, iter, fix=True):
        self.G.eval()

        img_save_dir = f'{self.save_dir}/GAN/img'
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if torch.cuda.is_available():
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if torch.cuda.is_available():
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)
        samples = (samples + 1) / 2
        utils.save_images(samples, f'{img_save_dir}/{epoch:03d}_{iter:04d}.png')

    def save(self):
        model_save_dir = f'{self.save_dir}/GAN/model'

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        torch.save(self.G.state_dict(), os.path.join(model_save_dir, '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(model_save_dir, '_D.pkl'))

        with open(os.path.join(model_save_dir, 'history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        model_save_dir = f'{self.save_dir}/GAN/model'

        self.G.load_state_dict(torch.load(os.path.join(model_save_dir, '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(model_save_dir, '_D.pkl')))