import torch
import torch.utils.data
import dataset
import time
import factory
import utils
import os

DEBUG = False
gpu = 0
lr = 2.5e-4
momentum = 0.9
weight_decay = 5e-4
K = 100
lr_factor = 0.2

lr_update_iter = 20000
ckpt_iter = 2000

ckpt_file = None #train from deeplab pretrained model, pretrained/deeplab.pth, download it from https://drive.google.com/uc?id=1Vi9mFuXk03GBbSV_3smjFA8S5-t3xj1h&export=download
# ckpt_file = 'PATH TO YOUR CHECKPOINT FILE'

utils.prepare()
print('running on gpu {}'.format(gpu))
print('debug {}, lr_factor {}, lr_update_iter {}'.format(DEBUG, lr_factor, lr_update_iter))

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

model = factory.make_siamese_model(ckpt_file).cuda()
optim = factory.make_optim(model, lr, momentum, weight_decay)

dataset = dataset.DAVIS2017(DEBUG)
trainloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle=True, num_workers=1)


st = time.time()
for ix, ((img1, anno1), (img2, anno2), (video,)) in enumerate(trainloader):
    model.train()
    img1, anno1, img2, anno2 = img1.to('cuda'), anno1.to('cuda'), img2.to('cuda'), anno2.to('cuda')
    loss = model.calc_loss(img1, anno1, img2, anno2, K, utils.get_max_label(video))
    optim.zero_grad()
    loss.backward()
    optim.step()
    print('{}/100000. {:.02f} seconds passed. current loss {:.05f}, lr {}, video {}'.format(ix+1, time.time() - st, loss.data, lr, video), flush=True)
    if ix % ckpt_iter == 0:
        #checkpoint
        utils.save_to_checkpoint(model, ix, DEBUG)

    if (ix+1) % lr_update_iter == 0:
        #adjust learning rate
        lr_before = lr
        lr = lr * lr_factor
        print('adjust learning rate .. form {} to {}'.format(lr_before, lr))
        optim = factory.make_optim(model, lr, momentum, weight_decay)

    if (DEBUG and ix % 50 == 0) or ix % 500 == 0 or (ix < 500 and ix % 100 == 0):
        model.eval()
        print('inference started.')
        utils.inference(model, K, ix, DEBUG)
        print('inference finished.')
