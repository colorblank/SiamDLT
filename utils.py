import torch
import os
import dataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

NAME = 'regular'
# N, class number
def stocastic_pooling(out, anno, K, N):
    assert(out.size(0) == 1) #only support batch size = 1
    d, h, w = out.size(1), out.size(2), out.size(3)
    #out (batch_size * d * h * w)
    #0~N, every class (anno == label) , random choose K pixels

    choosed_ij = []
    expand_labels = []
    for label in range(N+1):
        label_index_list = torch.nonzero(anno == label) # for example, size 6761 * 3
        if len(label_index_list) == 0:
            continue
        label_i_expend = torch.zeros((K, N+1)).to('cuda')
        label_i_expend[:, label] = 1.
        expand_labels.append(label_i_expend)

        choosed_ij.append(label_index_list[torch.randint(0, len(label_index_list), (K, )).long()])
    choosed_ij = torch.cat(choosed_ij, dim=0)
    expand_labels = torch.cat(expand_labels, dim=0)
    rand_i = choosed_ij[:, 1]
    rand_j = choosed_ij[:, 2]

    # rand_i = torch.randint(0, d, (K, )).long()
    # rand_j = torch.randint(0, h, (K, )).long()
    return out[:, :, rand_i, rand_j], rand_i, rand_j, expand_labels

def save_to_checkpoint(model, iter, debug=False):
    prefix = 'DEBUG-' if debug else ''
    name = '{}ckpt-iter-{:06}-{}.pth'.format(prefix, iter, NAME)
    print('saving checkpoint {} ...'.format(name))
    torch.save(model.state_dict(), os.path.join('checkpoint', name))
    print('checkpoint {} saved.'.format(name))

def inference(model, K, iter, debug=False):
    #test set && traing set && test set
    videos = ['dog', 'dog-gooses', 'blackswan'] if not debug else ['boat']
    frame_num_mapping = [60, 86, 50] if not debug else [75]
    assert len(videos) == len(frame_num_mapping)
    ds = dataset.DAVIS2017()
    with torch.no_grad():
        for idx in range(len(videos)):
            video = videos[idx]
            frame_num = frame_num_mapping[idx]
            inference_frame = [1, 9, frame_num-1] #2nd, 10th && last frame

            #first frame
            img1, anno1 = ds.get_img_anno_pair(video, 0)
            img1, anno1 = img1.to('cuda'), anno1.to('cuda')

            for frame_idx in inference_frame:
                img2, anno2 = ds.get_img_anno_pair(video, frame_idx)
                img2, anno2 = img2.to('cuda'), anno2.to('cuda')
                pred = model.inference_single(img1, anno1, img2, K)

                plt.clf()
                plt.subplot(3, 1, 2)
                plt.imshow(anno2.cpu().numpy())
                plt.subplot(3, 1, 1)
                plt.imshow(ds.get_original_img(video, frame_idx))
                plt.subplot(3, 1, 3)
                plt.imshow(pred)

                prefix = 'DEBUG-' if debug else ''
                # folder = os.path.join('result', '{}iter{:05}'.format('' if not debug else 'Debug-', iter))
                folder = os.path.join('result', '{}{}'.format(prefix, NAME))
                try:
                    os.mkdir(folder)
                except Exception:
                    pass
                plt.savefig(os.path.join(folder, '{}-{:03}-iter{:05}.png'.format(video, frame_idx, iter)))



video_max_label_map = {
'bear':1,
'bmx-bumps':2,
'boat':1,
'boxing-fisheye':3,
'breakdance-flare':1,
'bus':1,
'car-turn':1,
'cat-girl':2,
'classic-car':3,
'color-run':3,
'crossing':3,
'dance-jump':1,
'dancing':3,
'disc-jockey':3,
'dog-agility':1,
'dog-gooses':5,
'dogs-scale':4,
'drift-turn':1,
'drone':5,
'elephant':1,
'flamingo':1,
'hike':1,
'hockey':3,
'horsejump-low':2,
'kid-football':2,
'kite-walk':3,
'koala':1,
'lady-running':2,
'lindy-hop':8,
'longboard':5,
'lucia':1,
'mallard-fly':1,
'mallard-water':1,
'miami-surf':6,
'motocross-bumps':2,
'motorbike':3,
'night-race':2,
'paragliding':2,
'planes-water':2,
'rallye':1,
'rhino':1,
'rollerblade':1,
'schoolgirls':7,
'scooter-board':2,
'scooter-gray':2,
'sheep':5,
'skate-park':2,
'snowboard':2,
'soccerball':1,
'stroller':2,
'stunt':2,
'surf':3,
'swing':3,
'tennis':2,
'tractor-sand':3,
'train':4,
'tuk-tuk':3,
'upside-down':2,
'varanus-cage':1,
'walking':2
}


def get_max_label(video):
    try:
        num = video_max_label_map[video]
    except:
        num = 7
    return num
