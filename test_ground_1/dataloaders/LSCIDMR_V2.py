import pandas as pd
import numpy as np
import torch.utils.data
import time
import torchvision.transforms
import utils.augmentation

import numpy as np

def data_augmentation(image):
    mode = np.random.randint(0,7)
    if mode == 0:
        # original
        return image

    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image,axes= (1,2))
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image,axes= (1,2))
        return np.flip(image,axis=1)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2, axes= (1,2))
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2, axes= (1,2))
        return np.flip(image,axis=1)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3, axes= (1,2))
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3, axes=(1,2))
        return np.flip(image,axis=1)

class MLC_Dataset_16c(torch.utils.data.Dataset):


    def __init__(self, csv_file, root_dir, num_labels ,known_labels=0,transform=None,testing=False,tk_ratio=0.25):
        """
        Args:
            csv_file (string): Path to the csv file with annotations. 标签所在的文件。
            root_dir (string): Directory with all the images. 存放所有npy数据的文件夹。
            num_labels: 标签的数量
            known_labels: 可知标签的数量，用于部分标签已知的多标签分类。默认为0，代表全部位置未知
            transform (callable, optional): Optional transform to be applied on a sample. dataloader；
                里面传进来的变换文件。注意，这里最好只

        """
        self.tk_ratio = tk_ratio

        # ground truth
        self.labels_frame = pd.read_csv(csv_file)

        # img dir
        self.root_dir = root_dir

        # transform
        self.transform = transform
        self.testing = testing
        self.num_labels = num_labels
        self.known_labels = known_labels

        # for multiworkers
        self.start = 0
        self.end = len(self.labels_frame) # no need to -1

        # file_path_list
        self.file_path =self.root_dir+'/20190101.npy'
        self.whole_hsi =  np.load(self.file_path) # 读入整体的整幅数据
        self.ymd_str = '20190101'
        self.max = self.whole_hsi.reshape(16,-1).max(axis=1).reshape(16,1,1)
        self.min = self.whole_hsi.reshape(16,-1).min(axis=1).reshape(16,1,1)
        self.whole_hsi = (self.whole_hsi-self.min)/(self.max-self.min)*2-1 #整体进行归一化到[-1,1]


    def __len__(self):
        # get the length of data
        return len(self.labels_frame)

    def __getitem__(self, idx):
        """

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patch_name = self.labels_frame.iloc[idx, 0].replace('.png', '')
        patch_ymd = patch_name[0:8]
        loc_x,loc_y = int(patch_name[-2:])-1,int(patch_name[-5:-3])-1
        img_month = int(patch_name[4:6])-1
        #找出patch的具体的位置、月份

        # update data
        if patch_ymd != self.ymd_str: # 如果想要读进来的patch不在整个的大disk上就要重新读入一个新的大disk(npy文件)
            self.ymd_str = patch_ymd
            self.file_path = self.root_dir+'/'+patch_ymd+'.npy'
            self.whole_hsi = np.load(self.file_path)
            self.whole_hsi=(self.whole_hsi-self.min)/(self.max-self.min)*2-1 # 归一化到[-1,1]之间
            arr = self.whole_hsi.reshape((16,-1))
            #print(np.min(arr,axis=1))


        # 选一个小patch
        step = 60 #200 步长
        w = 300 #1000 长宽
        patch = self.whole_hsi[:, loc_y * step:loc_y * step + w, loc_x * step:loc_x * step + w]

        if not self.testing: # 不是训练数据使用数据增强
            patch = data_augmentation(patch)

        patch = torch.from_numpy(patch.copy())
        patch = torchvision.transforms.Resize(256)(patch)



        img_loc = [loc_x, loc_y]
        labels = self.labels_frame.iloc[idx, 1:]
        labels = np.array([labels])
        labels = labels.astype('float').reshape((-1))
        # print(labels.shape)
        image_id = self.labels_frame.iloc[idx, 0]
        sample = {'image': patch, 'labels': labels}


        if self.transform:
            hsi = self.transform(patch)
            labels = torch.Tensor(labels)


        caption = np.array(range(17))
        caption = caption * labels.numpy()
        caption[caption == 0] = 17
        caption.sort()
        caption = torch.LongTensor(caption)
        sample['length'] = labels.sum().to(torch.int64)  # for cnn-rnn
        sample['caption'] = caption  # for cnn-rnn/ the index of labels

        sample['image'] = hsi #图像数据
        sample['labels'] = labels #标签数据
        sample['imageIDs'] = str(image_id)
        sample['image_loc'] = img_loc
        sample['loc_num'] = (img_loc[0]) * 11 + (img_loc[1]) # 位置编号
        sample['month'] = img_month #数据所属月份

        return sample



"""
    下面是怎么取到一个dataloader的一个小demo
"""
if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # 所有数据（图像）的文件夹
    folder = '/data/zdxy/DataSets/small_whole'

    # 训练数据名及其标签
    train_csv = '/data/zdxy/DataSets/MLC_16c/check/multi_train_shuffled_checked.csv'

    # 验证数据名及其标签
    valid_csv = '/data/zdxy/DataSets/MLC_16c/check/multi_valid_checked.csv'

    # 测试数据名及其标签
    test_csv = '/data/zdxy/DataSets/MLC_16c/check/multi_test_checked.csv'

    train_trans = transforms.Compose([
        transforms.Resize(256),

    ])
    test_trans = transforms.Compose([
        transforms.Resize(256),
    ])

    # 下面是三个数据集
    train_dataset = MLC_Dataset_16c(train_csv, folder,
                                    num_labels=17,
                                    known_labels=0,
                                    transform=train_trans,  # torchvision.transforms.ToTensor(),
                                    testing=False,
                                    )

    valid_dataset = MLC_Dataset_16c(valid_csv, folder,
                                    num_labels=17,
                                    known_labels=0,
                                    transform=test_trans,  # torchvision.transforms.ToTensor(),
                                    testing=True,
                                    )

    test_dataset = MLC_Dataset_16c(test_csv, folder,
                                   num_labels=17,
                                   known_labels=0,
                                   transform=test_trans,  # torchvision.transforms.ToTensor(),
                                   testing=True
                                   )

    # 拿到三个dataloader
    # 这里有些参数需要你自己设置一下，比如batch_size,以及worker等等
    #
    # 由于硬盘性能的瓶颈，我在训练效果和读取效率上做了一点折中，dataset中一次性将很多数据读入，并在内存中分割；由于数据中存在重合部分，这样能省一点IO
    # 每一天的数据在内部是已经打乱过的，下面的shuffle强烈不建议设为True，会导致训练速度很慢
    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, batch_size= args.batch_size,shuffle=False, num_workers=workers,drop_last=drop_last,
                                   pin_memory=False) #,worker_init_fn=worker_init_fn
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size,shuffle=False, num_workers=workers,
                                   pin_memory=False)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,shuffle=False, num_workers=workers,
                                  pin_memory=False)

    # 然后下面就可以遍历了