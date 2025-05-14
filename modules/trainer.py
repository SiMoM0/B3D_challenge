import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cuda as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from modules.model import Modellone
from utils.avgmeter import AverageMeter
from utils.focal_loss import FocalLoss

from torch.utils.data import DataLoader
from Building3D.datasets import build_dataset

###### PARAMETERS ######
EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_DIM = 128
OPTIMIZER = 'adamw' # 'sgd'
SHOW_SAMPLE = False
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
        self.train_loader = DataLoader(building3d_dataset['train'], batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4, collate_fn=building3d_dataset['train'].collate_batch)
        print('Dataset size: ', len(self.train_loader.dataset))

        # import model
        self.model = Modellone(channels_in=3, channels_out=HIDDEN_DIM, big=True)

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
        weights = torch.tensor([0.1, 0.9]).float()
        self.ce_loss = nn.CrossEntropyLoss(weight=weights).to(self.device)
        #self.ce_loss = FocalLoss(alpha=weights, gamma=2).to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)
        # loss as dataparallel
        if self.n_gpus > 1:
            self.ce_loss = nn.DataParallel(self.ce_loss).cuda()
            self.mse_loss = nn.DataParallel(self.mse_loss).cuda()

        # optimizer
        if OPTIMIZER == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=1e-3,
                                               eps=1e-8,
                                               weight_decay=0.01)
        elif OPTIMIZER == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=1e-2,
                                             momentum=0.9,
                                             weight_decay=5e-4)
        else:
            raise ValueError(f"Optimizer {OPTIMIZER} not supported. Use 'adamw' or 'sgd'.")
        
        # scheduler
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

    def train(self):

        best_f1 = 0.0

        # train for n epochs
        for epoch in range(EPOCHS):
            # train for one epoch
            loss, closs, rloss, prec, rec, f1 = self.train_epoch(train_loader=self.train_loader,
                                            model=self.model,
                                            epoch=epoch)
            
            #print info
            print(f"\nEpoch: [{epoch+1}/{EPOCHS}] Loss: {loss:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}\n")

            if f1 > best_f1:
                best_f1 = f1
                # save model
                if self.n_gpus > 1:
                    torch.save(self.model.module.state_dict(), self.logdir + '/best_train.pth')
                else:
                    torch.save(self.model.state_dict(), self.logdir + '/best_train.pth')
                print(f"Model saved at epoch {epoch+1} with f1 {f1:.4f}")
                print('*'*50)
            
            # TODO: evaluate on validation set
                
    def train_epoch(self, train_loader, model, epoch):

        # empty the cache
        if self.gpu:
            torch.cuda.empty_cache()

        running_loss = AverageMeter()
        running_closs = AverageMeter()
        running_rloss = AverageMeter()
        running_prec = AverageMeter()
        running_rec = AverageMeter()
        running_f1 = AverageMeter()

        # switch to train mode
        model.train()
        
        for i, batch in enumerate(train_loader):
        #for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            pc = batch['point_clouds'].cuda()
            vertices = batch['wf_vertices'].cuda()
            edges = batch['wf_edges'].cuda()
            neighbors_emb = batch['neighbors_emb'].cuda()
            class_labels = batch['class_label'].cuda()
            off_labels = batch['offset'].cuda()

            pc = pc.permute(0, 2, 1)  # shape (B, C, N)
            neighbors_emb = neighbors_emb.permute(0, 2, 1)  # shape (B, K, N)

            # compute output
            output, offset = model(pc[:, :3], neighbors_emb)

            preds = output.argmax(dim=1)  # shape (B, N)

            # compute loss
            class_loss = self.ce_loss(output, class_labels)
            offset_loss = self.mse_loss(offset, off_labels)

            train_loss = class_loss + offset_loss

            # compute gradient and do optimizer step
            self.optimizer.zero_grad()
            if self.n_gpus > 1:
                idx = torch.ones(self.n_gpus).cuda()
                train_loss.backward(idx)
            else:
                train_loss.backward()
            
            # optimizer and scheduler step
            self.optimizer.step()
            #self.scheduler.step()

            # measure accuracy and record loss
            train_loss = train_loss.mean()
            # metrics
            tp = ((preds == 1) & (class_labels == 1)).sum().float()
            fp = ((preds == 1) & (class_labels == 0)).sum().float()
            fn = ((preds == 0) & (class_labels == 1)).sum().float()
            # acc = (preds == class_labels).sum().float() / preds.numel()
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)

            # update loss
            running_loss.update(train_loss.item(), pc.shape[0])
            running_closs.update(class_loss.item(), pc.shape[0])
            running_rloss.update(offset_loss.item(), pc.shape[0])
            # update tp = class 1 accuracy
            running_prec.update(precision.item(), pc.shape[0])
            running_rec.update(recall.item(), pc.shape[0])
            running_f1.update(f1.item(), pc.shape[0])

            if i % 100 == 0:
                print(f"Batch: [{i+1}/{len(train_loader)}] CE Loss: {running_closs.avg:.4f} | R Loss: {running_rloss.avg:.4f} | Total Loss: {running_loss.avg:.4f} | P: {running_prec.avg:.4f} | R: {running_rec.avg:.4f} | F1: {running_f1.avg:.4f}")

            if i % 100 == 0 and SHOW_SAMPLE:
                # plot the two point clouds with matplotlib
                colors = np.where(preds[0, :, None].cpu().numpy() == 0, [0, 0, 0], [1, 0, 0])
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(pc[0, 0, :].cpu().numpy(), pc[0, 1, :].cpu().detach().numpy(), pc[0, 2, :].cpu().detach().numpy(), c=colors, s=2)
                plt.title(f"Batch: [{i+1}/{len(train_loader)}]")
                plt.show()
                

            # write to tensorboard
            header = 'Train/'
            self.writer_train.add_scalar(header + 'CE Loss', class_loss, i + len(train_loader) * epoch)
            self.writer_train.add_scalar(header + 'R Loss', offset_loss, i + len(train_loader) * epoch)
            self.writer_train.add_scalar(header + 'Precision', precision, i + len(train_loader) * epoch)
            self.writer_train.add_scalar(header + 'Recall', recall, i + len(train_loader) * epoch)
            self.writer_train.add_scalar(header + 'F1', f1, i + len(train_loader) * epoch)
            # self.writer_train.add_scalar(header + 'Learning Rate', scheduler.get_lr()[0], i + len(train_loader) * epoch)

        return running_loss.avg, running_closs.avg, running_rloss.avg, running_prec.avg, running_rec.avg, running_f1.avg
