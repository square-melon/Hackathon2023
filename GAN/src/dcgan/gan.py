import os
import time
import numpy as np
import torch
from src.dcgan.generator import Generator
from src.dcgan.discriminator import Discriminator
from src.dataloader import ImageLoader
import torch.utils.data
from torch.autograd import Variable
from torchvision.utils import save_image
from src.progress.progress.bar import Bar as Bar
from src.utils import load_ckpt, save_ckpt
import tqdm
import shutil

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train(config, args):
    img_res = (config['img_res_w'], config['img_res_h'])
    pic_path = config['pic_searched_path']
    pic_path = os.path.join(pic_path, args.gan)
    pic_gen_path = config['pic_gen_path']
    pic_gen_path = os.path.join(pic_gen_path, args.gan)
    batch_size = config['dcgan']['batch_size']
    epoch_num = config['dcgan']['epoch_num']
    latent_dim = config['dcgan']['latent_size']
    pic_num = config['search_num']
    lr = config['dcgan']['lr']
    b1 = config['dcgan']['b1']
    b2 = config['dcgan']['b2']

    if os.path.exists(pic_gen_path):
        shutil.rmtree(pic_gen_path)
    os.makedirs(pic_gen_path)

    if not os.path.exists('checkpoint/dcgan'):
        os.makedirs('checkpoint/dcgan')

    adversarial_loss = torch.nn.BCELoss()
    generator = Generator(img_res, latent_dim)
    discriminator = Discriminator(img_res)
    epoch_start = 0

    if args.resume:
        gen_ckpt = load_ckpt('checkpoint/dcgan/gen.pt')
        dis_ckpt = load_ckpt('checkpoint/dcgan/dis.pt')
        generator.load_state_dict(gen_ckpt['state_dict'])
        discriminator.load_state_dict(dis_ckpt['state_dict'])
        epoch_start = min(gen_ckpt['epoch'], dis_ckpt['epoch'])
    else:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    optim_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optim_D = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))

    if args.resume:
        optim_G.load_state_dict(gen_ckpt['optimizer'])
        optim_D.load_state_dict(dis_ckpt['optimizer'])

    dataset = ImageLoader(pic_path=pic_path, img_res=img_res)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config['dcgan']['num_workers'])

    Tensor = torch.cuda.FloatTensor
        
    for ep in range(epoch_start, epoch_num):
        bar = Bar('>>>', fill='>', max=len(dataloader))
        print('>>> epoch:', ep)
        start = time.time()
        batch_time = 0
        for i, imgs in enumerate(dataloader):
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            real_imgs = Variable(imgs.type(Tensor))

            optim_G.zero_grad()

            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optim_G.step()

            optim_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optim_D.step()

            if (ep + 1) % 1000 == 0 and i == 0:
                save_image(gen_imgs.data[:25], os.path.join(pic_gen_path, 'ep%02d_gen.png' % (ep)), nrow=5, normalize=True)
        
            # update summary
            # if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

            bar.suffix = '[Batch {cur}/{total}] [D loss: {d_loss:.4}] [G loss: {g_loss:.4}] [batch: {batchtime:.4}ms] [Total: {ttl}] [ETA: {eta:}]' \
                .format(cur=i+1,
                        total=len(dataloader),
                        d_loss=d_loss.item(),
                        g_loss=g_loss.item(),
                        batchtime=batch_time * 10.0,
                        ttl=bar.elapsed_td,
                        eta=bar.eta_td
                        )
            bar.next()
        print()

        if (ep+1) % 20 == 0:
            save_ckpt(generator, optim_G, ep, 'checkpoint/dcgan/gen.pt')
            save_ckpt(discriminator, optim_D, ep, 'checkpoint/dcgan/dis.pt')

def gen(config):
    pic_gen_path = config['pic_gen_path']
    pic_gen_num = config['gen_num']
    latent_dim = config['dcgan']['latent_size']
    img_res = (config['img_res_w'], config['img_res_h'])

    if not os.path.exists(pic_gen_path):
        os.makedirs(pic_gen_path)

    ckpt = load_ckpt('checkpoint/dcgan/gen.pt')

    generator = Generator(img_res, latent_dim)
    generator.load_state_dict(ckpt['state_dict'])
    generator.cuda()
    generator.eval()
    Tensor = torch.cuda.FloatTensor
        
    for i in range(pic_gen_num):
        print('>>> gen', i)
        z = Variable(Tensor(np.random.normal(0, 1, (1, latent_dim))))
        gen_img = generator(z)
        save_image(gen_img.data[:25], os.path.join(pic_gen_path, 'gen%02d.jpg' % i), nrow=5, normalize=True)
