import torch.nn as nn
import torch
affine_par = True
import utils
import itertools
import performance

logger = performance.Logger()

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
            padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self,dilation_series,padding_series,NoLabels):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(2048,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)


    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out



class ResNet(nn.Module):
    def __init__(self, block, layers, NoLabels):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation__ = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],NoLabels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation_=dilation__))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,NoLabels):
        return block(dilation_series,padding_series,NoLabels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

class Deeplab(nn.Module):
    def __init__(self, block, num_labels):
        super(Deeplab, self).__init__()
        self.Scale = ResNet(block, [3, 4, 23, 3], num_labels)

    def forward(self, x):
        return self.Scale(x)

class Siamese(nn.Module):
    def __init__(self, deeplab):
        super(Siamese, self).__init__()
        self.deeplab = deeplab
        self.tconv = nn.ConvTranspose2d(21, 21, 3, 2) #hardcode the channel size ..
        self.ups = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mp = nn.MaxPool2d(2, 2)
        self.bn_adjust = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def get_decoder_params(self):
        return itertools.chain(self.tconv.parameters(), self.ups.parameters(), self.bn_adjust.parameters(), self.mp.parameters(), self.sigmoid.parameters())

    def forward(self, input):
        output = self.deeplab(input) # 1/8 resolution
        output = self.tconv(output)
        output = self.ups(output)
        return output

    #N, number of class
    def calc_loss(self, input1, anno1, input2, anno2, K = 100, N=1):
        batch_size = input1.size()[0]
        anno1, anno2 = self.mp(anno1), self.mp(anno2)
        output1 = self.forward(input1)
        output2 = self.forward(input2)

        #stocastic poolin spend about 90% of the time !!!!
        filtered_out_1, i_1, j_1, expand_labels_1 = utils.stocastic_pooling(output1, anno1, K, N) #1 * d * P
        filtered_out_2, i_2, j_2, expand_labels_2 = utils.stocastic_pooling(output2, anno2, K, N) #1 * d * Q

        P = filtered_out_1.size(2)
        Q = filtered_out_2.size(2)


        label1 = anno1[:, i_1, j_1]
        label2 = anno2[:, i_2, j_2]

        # print('label size', label1.size(), label2.size(), 'expect 1 * {}, 1 * {}'.format(P, Q))

        similarity = []
        gt = torch.zeros((batch_size, 1, P, Q)).to('cuda')
        for b in range(batch_size):
            similarity.append((filtered_out_1.transpose(1, 2)[b, :, :] @ filtered_out_2[b, :, :]).unsqueeze(0))
            gt[b, 0] = expand_labels_1 @ expand_labels_2.transpose(0, 1)

        # print('process by batch finished', flush=True)
        similarity = torch.cat(similarity, dim=0).unsqueeze(dim=1)

        # print('similarity size', similarity.size(), 'expect 1 * 1 * {} * {}'.format(P, Q))
        similarity = self.bn_adjust(similarity)

        similarity = self.sigmoid(similarity)

        return self.criterion(similarity, gt)

    #only inference single (not batch) frames
    def inference_single(self, input1, anno1, input2, K = 100, N = None):
        anno1 = self.mp(anno1.unsqueeze(dim=0))
        #about 14% time spend on forward
        output1 = self.forward(input1.unsqueeze(dim=0)) #
        output2 = self.forward(input2.unsqueeze(dim=0)).squeeze(dim=0) # d * h * w
        #about 84% time spend on calc instance number
        if N is None:
            N = torch.max(anno1).long()  # the instance number
        filtered_out_1, i_1, j_1, expand_label = utils.stocastic_pooling(output1, anno1, K, N) # 1 * d * P
        P = filtered_out_1.size(2)

        #pretend finished.
        output1 = output1.squeeze(dim=0) #d * h * w
        anno1 = anno1.squeeze(dim=0)
        filtered_out_1 = filtered_out_1.squeeze(dim=0) #d * K

        d, h, w = output2.size()
        similarity = filtered_out_1.transpose(0, 1) @ output2.view(d, -1) # P * hw
        similarity = similarity.view(1, 1, P, -1)
        similarity = self.bn_adjust(similarity).squeeze() # P * hw
        similarity = self.sigmoid(similarity)
        similarity = similarity.view(P, h*w)

        # print('similarity size', similarity.size(), 'expect {} * {}'.format(K , h * w))
        # print('expand label size', expand_label.size(), 'expect {} * {}'.format(N+1, K))
        similarity_by_label = expand_label.transpose(0,1) @ similarity # (N + 1) * (h * w)
        # print('similarity by label size' , similarity_by_label.size(), 'expect {} * {}'.format(N+1, h * w))
        pred_label = torch.argmax(similarity_by_label, dim=0).view(h, w)

        return pred_label

def make_deeplab_model(pretrained_path):
    model = Deeplab(Bottleneck, 21)
    if pretrained_path is not None:
        print('loading pretrained model from {} ...'.format(pretrained_path))
        saved_state_dict = torch.load(pretrained_path)
        model.load_state_dict(saved_state_dict)
    return model

def make_siamese_model():
    deeplab = make_deeplab_model('pretrained/deeplab.pth')
    model = Siamese(deeplab)
    return model


