import dataset
import torch

ds = dataset.DAVIS2017()
# print(ds.videos)
for video in ds.videos:
    anno = ds.get_anno(video, 0)
    max_label = torch.max(anno.to('cuda').long())
    max_label = max_label.detach().cpu().numpy()
    print(video, max_label)


