import numpy as np
from PIL import Image

mean = np.array([0.0, 0.0, 0.0])
std = np.array([0.0, 0.0, 0.0])

with open('train_list.txt') as f:
    lines = f.read().splitlines()
    for line in lines:
        path = line.split(' ')[0]
        ofilename = path.split('/')[-1]
        img = Image.open(path)
        img = np.array(img).astype('float32').transpose((2,0,1)) / 255.0
        meani = []
        stdi = []
        for i in range(3):
            meani.append(np.mean(img[i]))
            stdi.append(np.std(img[i]))
        mean += meani
        std += stdi

        print('mean=(%f,%f,%f) std=(%f,%f,%f)' % (np.mean(img[...,0]),np.mean(img[...,1]),np.mean(img[...,2]), np.std(img[...,0]), np.std(img[...,1]), np.std(img[...,2])))
        img_std = []
        for i in range(3):
            img_std.append((img[i] - meani[i]) / stdi[i])
        img_std = np.array(img_std).transpose((1,2,0))
        i = Image.fromarray(img_std.astype('uint8'))
        i.save('output/std_{}'.format(ofilename))

#        break

for i in range(3):
    mean[i] /= len(lines)
    std[i] /= len(lines)
print('Mean={0} std={1}'.format(mean, std))
