import torch
import dataset
import factory
import utils
import os
import performance

DEBUG = True
gpu = 1 if DEBUG else 2
# gpu = 0
K = 100
print('running on gpu {}'.format(gpu))
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

ckpt_file = 'checkpoint/DEBUG-ckpt-iter-001000-fast-3.pth'

model = factory.make_siamese_model(ckpt_file).cuda()
model.eval()

dataset = dataset.DAVIS2017()
utils.inference(model, K, iter, DEBUG)


# performance.Logger.start('abc')
# performance.Logger.log()

