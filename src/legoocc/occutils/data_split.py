import pickle as pkl
from random import shuffle

ori_path = './data/nuscenes/nuscenes_infos_train.pkl'
img_path = './data/nuscenes_infos_train_img55.pkl'
pts_path = './data/nuscenes_infos_train_pts55.pkl'

with open(ori_path, 'rb') as f:
    data = pkl.load(f)
infos = data['infos']
metadata = data['metadata']
shuffle(infos)
split = len(infos) // 2
infos_img = infos[:split]
infos_pts = infos[split:]
data_img = dict(infos=infos_img, metadata=metadata)
data_pts = dict(infos=infos_pts, metadata=metadata)
with open(img_path, 'wb') as f:
    pkl.dump(data_img, f)
with open(pts_path, 'wb') as f:
    pkl.dump(data_pts, f)