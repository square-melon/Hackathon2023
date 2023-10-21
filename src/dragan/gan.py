import os
import time
import numpy as np
import torch
from src.dragan.generator import Generator
from src.dragan.discriminator import Discriminator
from src.dataloader import ImageLoader
import torch.utils.data
from torch.autograd import Variable
import torch.autograd as autograd
from torchvision.utils import save_image
from src.progress.progress.bar import Bar as Bar
import tqdm
import shutil
from src.utils import load_ckpt, save_ckpt

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def compute_gradient_penalty(D, X):
    """Calculates the gradient penalty loss for DRAGAN"""
    # Random weight term for interpolation
    Tensor = torch.cuda.FloatTensor
    alpha = Tensor(np.random.random(size=X.shape))
    lambda_gp = 10

    interpolates = alpha * X + ((1 - alpha) * (X + 0.5 * X.std() * torch.rand(X.size()).cuda()))
    interpolates = Variable(interpolates, requires_grad=True)

    d_interpolates = D(interpolates)

    fake = Variable(Tensor(X.shape[0], 1).fill_(1.0), requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train(config, args):
    img_res = (config['img_res_w'], config['img_res_h'])
    pic_path = config['pic_searched_path']
    pic_path = os.path.join(pic_path, args.gan)
    pic_gen_path = config['pic_gen_path']
    pic_gen_path = os.path.join(pic_gen_path, args.gan)
    batch_size = config['dragan']['batch_size']
    epoch_num = config['dragan']['epoch_num']
    latent_dim = config['dragan']['latent_size']
    pic_num = config['search_num']
    lr = config['dragan']['lr']
    b1 = config['dragan']['b1']
    b2 = config['dragan']['b2']

    if os.path.exists(pic_gen_path):
        shutil.rmtree(pic_gen_path)
    os.makedirs(pic_gen_path)

    if not os.path.exists('checkpoint/dragan'):
        os.makedirs('checkpoint/dragan')

    adversarial_loss = torch.nn.BCELoss()
    generator = Generator(img_res, latent_dim)
    discriminator = Discriminator(img_res)
    epoch_start = 0

    if args.resume:
        gen_ckpt = load_ckpt('checkpoint/dragan/gen.pt')
        dis_ckpt = load_ckpt('checkpoint/dragan/dis.pt')
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config['dragan']['num_workers'])


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

            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data.cuda())
            gradient_penalty.backward()

            optim_D.step()

            if (ep + 1) % 40 == 0:
                save_image(gen_imgs.data[:25], os.path.join(pic_gen_path, 'ep%02d_gen%02d.png' % (ep+1, i)), nrow=5, normalize=True)
        
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
            save_ckpt(generator, optim_G, ep, 'checkpoint/dragan/gen.pt')
            save_ckpt(discriminator, optim_D, ep, 'checkpoint/dragan/dis.pt')

def gen(config):
    pic_gen_path = config['pic_gen_path']
    pic_gen_num = config['gen_num']
    latent_dim = config['dragan']['latent_size']

    if not os.path.exists(pic_gen_path):
        os.makedirs(pic_gen_path)

    ckpt = load_ckpt('checkpoint/dragan/gen.pt')
    generator = Generator()
    generator.load_state_dict(ckpt['state_dict'])
    generator.cuda()
    generator.eval()
    Tensor = torch.cuda.FloatTensor
        
    for i in range(pic_gen_num):
        print('>>> gen', i)
        z = Variable(Tensor(np.random.normal(0, 1, (1, latent_dim))))
        gen_img = generator(z)
        save_image(gen_img.data[:25], os.path.join(pic_gen_path, 'gen%02d.png' % i), nrow=5, normalize=True)
