import os


trainval = '/mnt/lustre/datatag/bixueting/deeproute/training/groundtruth'
test = '/mnt/lustre/datatag/bixueting/deeproute/testing/pointcloud'

train_txt = '/mnt/lustre/datatag/bixueting/deeproute/ImageSets/train.txt'
val_txt = '/mnt/lustre/datatag/bixueting/deeproute/ImageSets/val.txt'
trainval_txt = '/mnt/lustre/datatag/bixueting/deeproute/ImageSets/trainval.txt'
test_txt = '/mnt/lustre/datatag/bixueting/deeproute/ImageSets/test.txt'

with open(train_txt, 'w') as f:
    for path, dir, files in os.walk(trainval):
        for file in sorted(files)[: 18000]:
            i = file[:-4] + '\n'
            f.write(i)


with open(val_txt, 'w') as f:
    for path, dir, files in os.walk(trainval):
        for file in sorted(files)[18000: 20000]:
            i = file[:-4] + '\n'
            f.write(i) 


with open(trainval_txt, 'w') as f:
    for path, dir, files in os.walk(trainval): 
        for file in files:
            i = file[:-4] + '\n'
            f.write(i) 


with open(test_txt, 'w') as f:
    for path, dir, files in os.walk(test): 
        for file in files:
            i = file[:-4] + '\n'
            f.write(i) 