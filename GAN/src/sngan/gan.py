import os
import time
import numpy as np
import torch
import torch.nn as nn
from src.sngan.model import Generator, Discriminator
from src.dataloader import ImageLoader
import torch.utils.data
from torch.autograd import Variable
import torch.autograd as autograd
from torchvision.utils import save_image
from src.progress.progress.bar import Bar as Bar
import tqdm
import shutil
from src.utils import load_ckpt, save_ckpt

disc_iters = 5

def train(config, args):
    img_res = (config['img_res_w'], config['img_res_h'])
    pic_path = config['pic_searched_path']
    pic_path = os.path.join(pic_path, args.gan)
    pic_gen_path = config['pic_gen_path']
    pic_gen_path = os.path.join(pic_gen_path, args.gan)
    batch_size = config['sngan']['batch_size']
    epoch_num = config['sngan']['epoch_num']
    pic_num = config['search_num']
    lr = config['sngan']['lr']

    if os.path.exists(pic_gen_path):
        shutil.rmtree(pic_gen_path)
    os.makedirs(pic_gen_path)

    if not os.path.exists('checkpoint/sngan'):
        os.makedirs('checkpoint/sngan')

    adversarial_loss = torch.nn.BCELoss()
    generator = Generator(img_res[0])
    discriminator = Discriminator()
    epoch_start = 0

    if args.resume:
        gen_ckpt = load_ckpt('checkpoint/sngan/gen.pt')
        dis_ckpt = load_ckpt('checkpoint/sngan/dis.pt')
        generator.load_state_dict(gen_ckpt['state_dict'])
        discriminator.load_state_dict(dis_ckpt['state_dict'])
        epoch_start = min(gen_ckpt['epoch'], dis_ckpt['epoch'])

    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    optim_G  = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0,0.9))
    optim_D = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr, betas=(0.0,0.9))

    if args.resume:
        optim_G.load_state_dict(gen_ckpt['optimizer'])
        optim_D.load_state_dict(dis_ckpt['optimizer'])

    dataset = ImageLoader(pic_path=pic_path, img_res=img_res)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config['sngan']['num_workers'])

    Tensor = torch.cuda.FloatTensor

    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.99)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_G, gamma=0.99)
        
    for ep in range(epoch_start, epoch_num):
        bar = Bar('>>>', fill='>', max=len(dataloader))
        print('>>> epoch:', ep)
        start = time.time()
        batch_time = 0
        for i, imgs in enumerate(dataloader):
            real_imgs = Variable(imgs.type(Tensor))

            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            for _ in range(disc_iters):

                optim_G.zero_grad()
                optim_D.zero_grad()

                z = Variable(Tensor(np.random.randn(batch_size, img_res[0])))
                disc_loss = nn.ReLU()(1.0 - discriminator(real_imgs)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()

                disc_loss.backward()
                optim_D.step()

            optim_D.zero_grad()
            optim_G.zero_grad()

            z = Variable(Tensor(np.random.randn(batch_size, img_res[0])))

            # Loss measures generator's ability to fool the discriminator
            gen_imgs = generator(z)
            g_loss = -discriminator(gen_imgs).mean()

            g_loss.backward()
            optim_G.step()

            scheduler_d.step()
            scheduler_g.step()

            # Measure discriminator's ability to classify real from generated samples   

            if (ep + 1) % 40 == 0:
                save_image(gen_imgs.data, os.path.join(pic_gen_path, 'ep%02d_gen%02d.png' % (ep+1, i)), nrow=5, normalize=True)
        
            # update summary
            # if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

            bar.suffix = '[Batch {cur}/{total}] [D loss: {d_loss:.4}] [G loss: {g_loss:.4}] [batch: {batchtime:.4}ms] [Total: {ttl}] [ETA: {eta:}]' \
                .format(cur=i+1,
                        total=len(dataloader),
                        d_loss=disc_loss.item(),
                        g_loss=g_loss.item(),
                        batchtime=batch_time * 10.0,
                        ttl=bar.elapsed_td,
                        eta=bar.eta_td
                        )
            bar.next()
        print()

        if (ep+1) % 20 == 0:
            save_ckpt(generator, optim_G, ep, 'checkpoint/sngan/gen.pt')
            save_ckpt(discriminator, optim_D, ep, 'checkpoint/sngan/dis.pt')

def gen(config):
    pic_gen_path = config['pic_gen_path']
    pic_gen_num = config['gen_num']
    latent_dim = config['sngan']['latent_size']
    img_res = (config['img_res_w'], config['img_res_h'])

    if not os.path.exists(pic_gen_path):
        os.makedirs(pic_gen_path)

    ckpt = load_ckpt('checkpoint/sngan/gen.pt')
    generator = Generator()
    generator.load_state_dict(ckpt['state_dict'])
    generator.cuda()
    generator.eval()
    Tensor = torch.cuda.FloatTensor
        
    for i in range(pic_gen_num):
        print('>>> gen', i)
        z = Variable(Tensor(np.random.normal(0, 1, (1, img_res[0]))))
        gen_img = generator(z)
        save_image(gen_img.data, os.path.join(pic_gen_path, 'gen%02d.png' % i), normalize=True)
