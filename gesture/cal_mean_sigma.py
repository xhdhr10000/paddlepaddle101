import numpy as np
from PIL import Image
import sys

mean = 0
std = 0

filename = 'train_list.txt'
if len(sys.argv) > 1: filename = sys.argv[1]

with open(filename) as f:
    lines = f.read().splitlines()
    for line in lines:
        path = line.split(' ')[0]
        ofilename = path.split('/')[-1]
        img = Image.open(path).convert('L')
        img = np.array(img).astype('float32') / 255.0

        print('mean=%f std=%f' % (np.mean(img), np.std(img)))
        mean += np.mean(img)
        std += np.std(img)
        img = 1 - img
        """
        for i in range(len(img)):
            for j in range(len(img[i])):
                if img[i][j] < 0.1: img[i][j] = 0
                    """
        img = (img - np.mean(img)) / np.std(img)

        i = Image.fromarray(img * 255).convert('RGB')
        i.save('output/std_{}'.format(ofilename))

#        break

mean /= len(lines)
std /= len(lines)
print('Mean={0} std={1}'.format(mean, std))
