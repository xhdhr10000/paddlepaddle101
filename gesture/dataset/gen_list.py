import os

# train
with open('train_list.txt', 'w') as f:
	train_dir = 'train'
	for d in os.listdir(train_dir):
		path = os.path.join(train_dir, d)
		for image in os.listdir(path):
			f.write('%s %s\n' % (os.path.join(path, image), d))

# test
with open('val_list.txt', 'w') as f:
	val_dir = 'test'
	for d in os.listdir(val_dir):
		path = os.path.join(val_dir, d)
		for image in os.listdir(path):
			f.write('%s %s\n' % (os.path.join(path, image), d))
