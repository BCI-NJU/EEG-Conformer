'''
ExP class.
The training logic.
'''


import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from models import Conformer
from torch.backends import cudnn
from torchsummary import summary
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import scipy.io
import sys

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # arrange GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus)) # choose GPUs

class ExP():
    def __init__(self, nsub, config=None):
        super(ExP, self).__init__()
        if config is None:
            print("no customized config, using default config")
            self.batch_size = 72
            self.n_epochs = 500
            self.lr = 0.0002
            self.b1 = 0.5
            self.b2 = 0.999
            self.nSub = nsub
        else:
            self.batch_size=config['batch_size']
            self.n_epochs=config['n_epochs']
            self.lr=config['lr']
            self.b1=config['b1']
            self.b2=config['b2']
            self.nSub=nsub


        self.start_epoch = 0
        self.root = './Datasets/lyh_dataset'
        # res_path = "./Results/log_subject%d.txt" % self.nSub
        
        self.config = config
        if self.config != None:
            res_path = config["res_path"]
        
        dir_name = os.path.dirname(res_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        result_write = open(os.path.join(res_path, "exp_log.txt"), "w")
        self.log_write = open(os.path.join(res_path, "exp_log.txt"), "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = Conformer(config=self.config).cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        summary(self.model, (1, 14, 1000))
        
#         with open('summary.txt', 'w') as f:
#             original_stdout = sys.stdout 
#             sys.stdout = f

#             summary(self.model, (1, 22, 1000))
#             sys.stdout = original_stdout

    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):
        # print("In interaug.")
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            # cls_idx = np.where(label == cls4aug + 1)
            cls_idx = np.where(label == cls4aug)
            tmp_data = timg[cls_idx]
            tmp_data = tmp_data.reshape(tmp_data.shape[0], -1, *tmp_data.shape[-2:])
            tmp_label = label[cls_idx]
            # print(timg.shape) # (288, 1, 22, 1000)
            # print(tmp_data.shape) # (72, 1, 22, 1000)
            # print(label.shape) # (288, 1)
            # print(tmp_label.shape) # (72,)

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 14, 1000))
            # print(f"tmp_aug_data.shape = {tmp_aug_data.shape}")
            # drf: get 8 slices of 8 random data from tmp_data, from the same time period idx
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    # print(f"ri, rj: {ri}, {rj}")
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        # aug_label = torch.from_numpy(aug_label-1).cuda()
        aug_label = torch.from_numpy(aug_label).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_test_data(self):
        """
        This function is used for testing the model's performance
        will load the test_set.mat files 
        Returns:
        (X_test, y_test)
        """

        test_set = scipy.io.loadmat('Datasets/lyh_dataset/test_set.mat')

        X_test = test_set['X_test']
        y_test = test_set['y_test']


        return X_test, y_test 
    
    def get_source_data(self):
        """
        This function is used to make train set and test set from four raw EEG data in .npy format.
        Also, the (X_test, y_test) will be saved in .mat format for future evaluations.
        Returns:
        (X_train, y_train, X_test, y_test)

        """

        # get all the data first
        cur_path = os.getcwd()
        print(cur_path)
        left_raw = np.load('Datasets/lyh_dataset/left_processed.npy') # label: 0
        right_raw = np.load('Datasets/lyh_dataset/right_processed.npy') # label: 1
        leg_raw = np.load('Datasets/lyh_dataset/left_processed.npy') # label: 2
        nothing_raw = np.load('Datasets/lyh_dataset/nothing_processed.npy') # label: 3
        eeg_raw = [left_raw, right_raw, leg_raw, nothing_raw]

        X_tot = []
        y_tot = []

        for i in range(4):
            tmp = eeg_raw[i].reshape(15, 300, -1) # (15, 30_0000) => (15, 300, 1000)
            tmp = tmp[:14, :, :] # filter the channels, only need the first 14 channels
            X_raw = tmp.transpose((1, 0, 2)) # (14, 300, 1000) => (300, 14, 1000)
            y_raw = np.array([i for j in range(300)]) # (300,) value = label
            X_tot.append(X_raw)
            y_tot.append(y_raw)

        X_tot = np.concatenate(X_tot)
        y_tot = np.concatenate(y_tot)

        # print(X_tot.shape, y_tot.shape) # (1200, 14, 1000), (1200,)
    

        self.train_data = X_tot # (1200, 14, 1000)
        self.train_data = np.expand_dims(self.train_data, axis=1) # (1200, 1, 14, 1000)
        self.train_label = y_tot.reshape(1200, 1) # (1200, 1)
        
        self.allData = self.train_data # (1200, 1, 14, 1000)
        self.allLabel = self.train_label.squeeze() # (1200, )

        shuffle_num = np.random.permutation(len(self.allData))
        # print(f"Shuffle num {shuffle_num}")
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        # split the dataset
        X_train, X_test, y_train, y_test = train_test_split(self.allData, self.allLabel, train_size=0.7,
                                                                random_state=None, shuffle=True)
        
        # save the (X_test, y_test) into .mat files for further use
        
        test_set = {
            'X_test': X_test,
            'y_test': y_test
        }

        scipy.io.savemat('./Datasets/lyh_dataset/test_set.mat', test_set)


        # standardize
        # target_mean = np.mean(self.allData)
        # target_std = np.std(self.allData)
        # self.allData = (self.allData - target_mean) / target_std
        # self.testData = (self.testData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        # print(self.allData.shape) # (288, 1, 22, 1000)
        # print(self.allLabel.shape) # (288, 1)
        # print(self.testData.shape)
        # print(self.testLabel.shape)
        # print(self.testLabel)
        return X_train, y_train, X_test, y_test

    def train(self):

        # some trackable history
        train_acc_list = []
        test_acc_list = []
        train_loss_list = []
        test_loss_list = []

        n_track_epochs = self.n_epochs // 100 # track the loss & acc every n_track_epochs epochs.

        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        # label = torch.from_numpy(label - 1)
        # print(img.shape)
        # print(label.shape)

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)
        # test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        for e in range(self.n_epochs):
            # in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # data augmentation
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                # print(aug_data.shape)
                # print(aug_label.shape)
                # print(self.allData.shape)
                # print(self.allLabel.shape)
                
                # print(label.shape)
                # print(aug_label.shape)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))
                
                outputs = self.model(img)
                # tok, outputs = self.model(img)

                loss = self.criterion_cls(outputs, label) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            # out_epoch = time.time()

            # test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                # Tok, Cls = self.model(test_data)
                Cls = self.model(test_data)


                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc)
                
                if (e + 1) % n_track_epochs == 0:
                    train_acc_list.append(train_acc)
                    test_acc_list.append(acc)
                    train_loss_list.append(loss.detach().cpu().numpy())
                    test_loss_list.append(loss_test.detach().cpu().numpy())

                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred

        dir_name = "Models"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        torch.save(self.model.module.state_dict(), 'Models/lyh_dataset/model_sub%d.pth'%self.nSub)
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        # draw the plot
        if train_acc_list:
            plt.figure()
            x = [i for i in range(1, len(train_acc_list) + 1)]
            plt.plot(x, train_acc_list, label="acc_train")
            plt.plot(x, test_acc_list, label="acc_test")

            plt.legend()

            plt.xlabel("epoch")
            plt.ylabel("accuracy")

            plt.savefig(os.path.join(self.config['res_path'], "acc.png"))


            plt.figure()
            x = [i for i in range(1, len(train_loss_list) + 1)]
            plt.plot(x, train_loss_list, label="loss_train")
            plt.plot(x, test_loss_list, label="loss_test")

            plt.legend()

            plt.xlabel("epoch")
            plt.ylabel("loss")

            plt.savefig(os.path.join(self.config['res_path'], "loss.png"))

        return bestAcc, averAcc, Y_true, Y_pred
        # writer.close()