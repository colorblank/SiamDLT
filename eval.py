import dataset
import factory
import utils
import os

DEBUG = True
gpu = 1
K = 100
print('running on gpu {}'.format(gpu))
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

ckpt_file = 'PATH TO YOUR CHECKPOINT FILE'

model = factory.make_siamese_model(ckpt_file).cuda()
model.eval()

dataset = dataset.DAVIS2017(DEBUG)
utils.inference(model, K, 'unknown', DEBUG)

