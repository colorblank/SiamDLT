import torch
# from deeplab_resnet import make_deeplab_model
import deeplab_resnet
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# model = make_deeplab_model('pretrained/deeplab.pth')
# model.eval()
# saved_state_dict = torch.load('pretrained/MS_DeepLab_resnet_trained_VOC.pth')
# saved_state_dict = torch.load()
# model.load_state_dict(saved_state_dict)
# model.cuda()

# model = deeplab_resnet.make_deeplab_model('pretrained/MS_DeepLab_resnet_trained_VOC.pth')

# model = deeplab_resnet.make_deeplab_model('pretrained/deeplab.pth')

model = deeplab_resnet.make_siamese_model()


# model = deeplab_resnet.Res_Deeplab(21)
# model.eval()
# saved_state_dict = torch.load('pretrained/MS_DeepLab_resnet_trained_VOC.pth')
# model.load_state_dict(saved_state_dict)
model.eval()
model.cuda()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

img_file = '/home/cly/datacenter/DAVIS/JPEGImages/480p/dog/00000.jpg'
image = np.asarray(Image.open(img_file).convert('RGB'), np.float32)
image-= IMG_MEAN
image = image.transpose((2, 0, 1))
input = torch.from_numpy(image).unsqueeze(dim=0).to('cuda')

# print('input size', input.size())

output_list = [model(input)]
# for each in output_list:
#     print('in demo', each.size())

for idx in range(len(output_list)):
    interp = nn.UpsamplingBilinear2d(size=(480, 854))
    output = interp(output_list[idx]).detach().squeeze().cpu().numpy()
    output = output.transpose(1, 2, 0)

    print('before argmax', output.shape)
    output = np.argmax(output, axis=2)
    print('after argmax', output.shape)

    plt.subplot(len(output_list), 1, idx + 1)
    plt.imshow(output)


plt.show()
plt.savefig('result/demo2.png')

exit(0)
output = model(input)
print('finished, outputsize ', output.size())
# for each in output_list:
#     print('in demo', each.size())

interp = nn.UpsamplingBilinear2d(size=(480, 854))
output = interp(output).detach().squeeze().cpu().numpy()
output = output.transpose(1, 2, 0)

print('before argmax', output.shape)
output = np.argmax(output, axis=2)
print('after argmax', output.shape)

plt.subplot(1, 1, 1)
plt.imshow(output)
plt.show()
plt.savefig('result/demo.png')

print('load model successful')
