import os
import shutil

import scipy.io as scio

_image_labels = '/Users/Konyaka/Downloads/flowers102/imagelabels.mat'
_set_id = '/Users/Konyaka/Downloads/flowers102/setid.mat'
source = '/Users/Konyaka/Downloads/flowers102/jpg/'
dst = '/Users/Konyaka/Downloads/flowers102/dataset_split/'


def mkdir(path):
    path = path.strip()
    path = path.rstrip("/")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)
        return True
    else:
        return False


def classify(set, dst, labels):
    for n, id in enumerate(set):
        cls = labels[id - 1]
        filename = 'image_%05d.jpg' % id
        mkdir('%s/%d' % (dst, cls))
        shutil.copy(source + filename, '%s/%d/%s' % (dst, cls, filename))
        print(n + 1)


image_labels = scio.loadmat(_image_labels)
set_id = scio.loadmat(_set_id)

classify(set_id['tstid'][0], dst + 'train/', image_labels['labels'][0])
classify(set_id['valid'][0], dst + 'valid/', image_labels['labels'][0])
classify(set_id['trnid'][0], dst + 'test/', image_labels['labels'][0])
