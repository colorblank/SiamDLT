import torch
import deeplab_resnet

def make_siamese_model(resume=None):
    deeplab_pretrained = 'pretrained/deeplab.pth' if resume is None else None
    deeplab = deeplab_resnet.make_deeplab_model(deeplab_pretrained)
    model = deeplab_resnet.Siamese(deeplab)
    if resume is not None:
        print('loading pretrained model from {} ...'.format(resume))
        saved_state_dict = torch.load(resume)
        model.load_state_dict(saved_state_dict)
        print('pretrained model loaded.')
    return model

def make_optim(model, lr, momentum, weight_decay):
    print('learning rate {}, momentum {}, weiht_decay {}'.format(lr, momentum, weight_decay))
    optim = torch.optim.SGD([
        {'params': filter(lambda p: p.requires_grad, model.deeplab.parameters()), },
        {'params': filter(lambda p: p.requires_grad, model.get_decoder_params()), 'lr': lr*2, 'weight_decay': 0}
    ], lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optim

