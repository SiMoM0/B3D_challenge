import yaml
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import torch
import torch.nn as nn
import torch.backends.cuda as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from scipy.spatial import cKDTree as KDTree


from modules.model import Modellone

from torch.utils.data import DataLoader
from Building3D.datasets import build_dataset

###### PARAMETERS ######
EPOCHS = 10
##########################

class Trainer():
    def __init__(self, dataset_config, logdir) -> None:
        # parameters
        self.dataset_config = dataset_config
        self.logdir = logdir

        # tensorboard
        self.writer_train = SummaryWriter(log_dir=self.logdir + '/tensorboard/train',
                                    purge_step=0,
                                    flush_secs=30)
        self.writer_valid = SummaryWriter(log_dir=self.logdir + '/tensorboard/valid',
                                          purge_step=0,
                                          flush_secs=30)

        # build dataset
        building3d_dataset = build_dataset(dataset_config.Building3D)

        # create dataloader
        self.train_loader = DataLoader(building3d_dataset['train'], batch_size=1, shuffle=False, drop_last=True, num_workers=4, collate_fn=building3d_dataset['train'].collate_batch)
        print('Dataset size: ', len(self.train_loader.dataset))

        # import model
        self.model = Modellone(channels_in=6, channels_out=64)

        # count number of parameters
        print('Total params: ', sum(p.numel() for p in self.model.parameters()))

        # GPU ?
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()
            self.model.cuda()

        # loss
        self.ce_loss = nn.CrossEntropyLoss().to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)
        # loss as dataparallel
        if self.n_gpus > 1:
            self.ce_loss = nn.DataParallel(self.ce_loss).cuda()
            self.mse_loss = nn.DataParallel(self.mse_loss).cuda()

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=1e-3,
                                         momentum=0.9,
                                         weight_decay=5e-4)
        
        # scheduler
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

    def train(self):

        # train for n epochs
        for epoch in range(EPOCHS):
            # train for one epoch
            closs, rloss = self.train_epoch(train_loader=self.train_loader,
                                            model=self.model,
                                            epoch=epoch)
            
            #print info
            print(f"Epoch: [{epoch+1}/{EPOCHS}] CE Loss: {closs:.4f} | R Loss: {rloss:.4f}")
            
            # TODO: evaluate on validation set
                
    def train_epoch(self, train_loader, model, epoch):

        # empty the cache
        if self.gpu:
            torch.cuda.empty_cache()

        loss = 0

        # switch to train mode
        model.train()
        
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            pc = batch['point_clouds'].cuda()
            vertices = batch['wf_vertices'].cuda()
            edges = batch['wf_edges'].cuda()

            # compute neighbors (TODO: move in dataloader)
            kdtree = KDTree(pc.cpu().numpy()[0, :, :3])
            _, neighbors_emb = kdtree.query(pc.cpu().numpy()[0, :, :3], k=10)
            neighbors_emb = torch.from_numpy(neighbors_emb).cuda().view(1, -1, 10)

            # compute labels for each point
            label = torch.zeros(pc.shape[1], dtype=torch.long).cuda()
            distances = torch.cdist(pc[:, :, :3], vertices[:, :, :3])
            distances = distances.min(dim=2)[0]  # shape (1, N), values in [0, 1]
            label = (distances < 0.1).long()  # shape (1, N), values in {0, 1}
            label = label.cuda().view(1, -1)

            pc = pc.permute(0, 2, 1)  # shape (B, C, N)
            neighbors_emb = neighbors_emb.permute(0, 2, 1)  # shape (B, K, N)

            # compute output
            output, offset = model(pc[:, :6], neighbors_emb)

            # compute loss
            loss = self.ce_loss(output, label)
            # TODO: regression loss

            loss = loss

            # compute gradient and do optimizer step
            self.optimizer.zero_grad()
            if self.n_gpus > 1:
                idx = torch.ones(self.n_gpus).cuda()
                loss.backward(idx)
            else:
                loss.backward()
            
            # optimizer and scheduler step
            self.optimizer.step()
            #self.scheduler.step()

            # measure accuracy and record loss
            loss = loss.mean()

            #print(f"Batch: [{i+1}/{len(train_loader)}] Loss: {loss:.4f}")

            # update loss
            loss += (loss.item() / pc.shape[0])

            if i % 100 == 0:
                print(f"Batch: [{i+1}/{len(train_loader)}] Loss: {loss:.4f}")

            # write to tensorboard
            header = 'Train/'
            self.writer_train.add_scalar(header + 'CE Loss', loss, i + len(train_loader) * epoch)
            # self.writer_train.add_scalar(header + 'Accuracy', acc.avg, i + len(train_loader) * epoch)
            # self.writer_train.add_scalar(header + 'IoU', iou.avg, i + len(train_loader) * epoch)
            # self.writer_train.add_scalar(header + 'Learning Rate', scheduler.get_lr()[0], i + len(train_loader) * epoch)

        return loss, 0
