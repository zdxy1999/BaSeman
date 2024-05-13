import argparse,math,time,warnings,copy, numpy as np, os.path as path 
import torch, torch.nn as nn, torch.nn.functional as F
from pdb import set_trace as stop
from tqdm import tqdm
from models.utils import custom_replace
import random


class DataPrefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != 'meta' and isinstance(self.batch[k],torch.Tensor):
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

def left_region_project(tensor):
    return torch.log(10*(tensor+1e-12)/(1-(10)*(tensor+1e-12)))

def right_region_project(tensor):
    return torch.log((1/1.9)*(tensor+0.9-1e-12)/(1-(1/1.9)*(tensor+0.9-1e-12)))

def reproject(tensor):
    tensor[tensor < 0.05] = left_region_project( tensor[tensor< 0.05])
    tensor[tensor >=0.05] = right_region_project(tensor[tensor>=0.05])
    return torch.sigmoid(tensor)

def run_epoch(args,model,data,optimizer,epoch,desc,train=False,warmup_scheduler=None,device=None):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    # pre-allocate full prediction and target tensors
    all_predictions = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_targets = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_masks = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_image_ids = []

    max_samples = args.max_samples

    batch_idx = 0
    loss_total = 0
    unk_loss_total = 0

    pbar = tqdm(data,mininterval=0.5,desc=desc,leave=False,ncols=120)
    month_pred = None
    for batch in pbar:
        if batch_idx == max_samples:
            break

        labels = batch['labels'].float()
        images = batch['image'].float()
        mask = batch['mask'].float()
        #zdxy
        img_loc = batch['image_loc']
        month_num = batch['month'].view(labels.shape[0],1)
        loc_num = batch['loc_num'].view(labels.shape[0],1)
        unk_mask = custom_replace(mask, 1, 0, 0)
        # all_image_ids += batch['imageIDs']
        # caption = batch['caption'] #for cnn_rnn
        # length = batch['length'] #for cnn_rnn

        #print(month_num.device)
        month_label  = torch.zeros(labels.shape[0],12).scatter_(1,month_num,1)
        loc_label = torch.zeros(labels.shape[0], 286).scatter_(1, loc_num, 1)

        month_all = torch.from_numpy(np.array(range(12))).view(12, 1)
        loc_all = torch.from_numpy(np.array(range(286))).view(286, 1)
        month_label_all = torch.zeros(12,12).scatter_(1,month_all,1)
        loc_label_all = torch.zeros(286, 286).scatter_(1, loc_all, 1)


        
        mask_in = mask.clone()

        if args.model=='ctran':
            if train:
                pred,int_pred,attns,month_pred,loc_pred = model(images.cuda(),mask_in.cuda(),img_loc,month_num,loc_num)
            else:
                with torch.no_grad():
                    pred,int_pred,attns,month_pred,loc_pred = model(images.cuda(),mask_in.cuda(),img_loc,month_num,loc_num)
        elif args.model=='split':
            if train:
                pred,int_pred,attns,month_pred,loc_pred = model(images.cuda(),mask_in.cuda(),img_loc,month_num,loc_num)
            else:
                with torch.no_grad():
                    pred,int_pred,attns,month_pred,loc_pred = model(images.cuda(),mask_in.cuda(),img_loc,month_num,loc_num)
        elif args.model == 'ctran_16c':
            if train:
                pred, int_pred, attns,month_pred,loc_pred = model(images.cuda(non_blocking=True), mask_in.cuda(non_blocking=True), img_loc, month_num, loc_num)
            else:
                with torch.no_grad():
                    pred, int_pred, attns,month_pred,loc_pred = model(images.cuda(non_blocking=True), mask_in.cuda(non_blocking=True), img_loc, month_num, loc_num)
        elif args.model == 'split_16c':
            if train:
                pred, int_pred, attns ,month_pred,loc_pred= model(images.cuda(non_blocking=True), mask_in.cuda(non_blocking=True), img_loc, month_num, loc_num)
            else:
                with torch.no_grad():
                    pred, int_pred, attns,month_pred,loc_pred = model(images.cuda(non_blocking=True), mask_in.cuda(non_blocking=True), img_loc, month_num, loc_num)
        elif args.model == 'together':
            if train:
                pred, int_pred, attns,month_pred,loc_pred  = model(images.cuda(non_blocking=True), mask_in.cuda(non_blocking=True), img_loc, month_num, loc_num)
            else:
                with torch.no_grad():
                    pred, int_pred, attns, month_pred,loc_pred  = model(images.cuda(non_blocking=True), mask_in.cuda(non_blocking=True), img_loc, month_num, loc_num)
        elif args.model == 'mc16':
            if train:
                pred, int_pred, attns,month_pred,loc_pred = model(images.cuda(non_blocking=True), mask_in.cuda(non_blocking=True), img_loc, month_num, loc_num)
            else:
                with torch.no_grad():
                    pred, int_pred, attns,month_pred,loc_pred = model(images.cuda(non_blocking=True), mask_in.cuda(non_blocking=True), img_loc, month_num, loc_num)
        elif args.model == 'cnn_rnn':
            if train:
                pred = model(images.cuda(),caption.cuda())
            else:
                with torch.no_grad():
                    pred = model.sample(images.cuda())
        elif args.model == 'add_gcn':
            if train:
                pred,_ = model(images.cuda(non_blocking=True))
            else:
                with torch.no_grad():
                    pred, _ = model(images.cuda(non_blocking=True))
        elif args.model == 'q2l':
            if train:
                pred = model(images.cuda(non_blocking=True))
            else:
                with torch.no_grad():
                    pred = model(images.cuda(non_blocking=True))
        elif args.model == 'original':
            if train:
                pred,int_pred,attns = model(images.cuda(),mask_in.cuda())
            else:
                with torch.no_grad():
                    pred,int_pred,attns = model(images.cuda(),mask_in.cuda())
        elif args.model == 'ssnet':
            if train:
                _,pred, month_pred, loc_pred = model(images.cuda(),month_num,loc_num)
            else:
                with torch.no_grad():
                    _,pred, month_pred, loc_pred = model(images.cuda(),month_num,loc_num)
        elif args.model == 'ac':
            if train:
                pred, similarLoss, candidates = model(images.cuda(),epoch,month_num,loc_num)
            else:
                with torch.no_grad():
                    pred, similarLoss, candidates = model(images.cuda(),200,month_num,loc_num)
        elif args.model == 's2net':
            if train:
                label_emb, spat_logits, spec_logits, overall_logits, ensemble_logits = model(images.cuda())
            else:
                with torch.no_grad():
                    label_emb, spat_logits, spec_logits, overall_logits,  ensemble_logits = model(images.cuda())
        elif args.model == 'tsformer':
            if train:
                pred = model(images.cuda())
            else:
                with torch.no_grad():
                    pred,_ = model(images.cuda())
        elif args.model == 'ida':
            if train:
                pred,_ = model(images.cuda())
            else:
                with torch.no_grad():
                    pred,_ = model(images.cuda())
        else:
            if train:
                pred = model(images.cuda())
            else:
                with torch.no_grad():
                    pred = model(images.cuda())


        month_loss = torch.tensor(0)
        loc_loss = torch.tensor(0)


        '''
        if args.soft_label:
            if train:
                labels[:,-9:] = reproject(labels[:,-9:])

            #pred[:,-9:] = reproject(torch.sigmoid(pred[:,-9:]))
            #pred[:,:-9] = torch.sigmoid(pred[:,:-9])

            pred = torch.sigmoid(pred)

        else:
            pred = torch.sigmoid(pred)
        '''
        if(args.model!='s2net'):
            loss = F.binary_cross_entropy_with_logits(pred.view(labels.size(0),-1),labels.cuda().detach(),reduction='none')

        if not (('ssnet' not in args.model) and ('ctran' not in args.model) and ('split' not in args.model) and ('mc' not in args.model)):
            if args.use_month:
                if ('mc' in args.model) and (month_pred.shape[0] > 12):
                    month_pred = month_pred[:12, :]
                #month_loss = F.binary_cross_entropy(month_pred.view(labels.size(0), -1), month_label.cuda(), reduction='none').sum()
                month_loss = F.binary_cross_entropy(month_pred.view(12, -1), month_label_all.cuda(), reduction='none').sum()/12
            if args.use_loc:
                if ('mc' in args.model) and (loc_pred.shape[0] > 286):
                   loc_pred = loc_pred[:286, :]
                #loc_loss = F.binary_cross_entropy(loc_pred.view(labels.size(0), -1), loc_label.cuda(), reduction='none').sum()
                loc_loss = F.binary_cross_entropy(loc_pred.view(286, -1), loc_label_all.cuda(), reduction='none').sum()/286
            #loss = F.binary_cross_entropy(pred.view(labels.size(0), -1), labels.cuda(), reduction='none')

        if args.loss_labels == 'unk':
            # only use unknown labels for loss
            loss_out = (unk_mask.cuda()*loss).mean()
        else:
            # use all labels for loss
            #print(type(month_loss),type(loc_loss))
            if (args.model != 's2net'):
                loss_out = loss.sum()/args.batch_size+0.5*month_loss/100+0.5*loc_loss/100
        if args.model == 'ac':
            loss_cnn = F.binary_cross_entropy_with_logits(candidates[:,0,:].view(labels.size(0),-1),labels.cuda().detach(),reduction='none').sum()/args.batch_size
            loss_trans = F.binary_cross_entropy_with_logits(candidates[:, 1, :].view(labels.size(0),-1), labels.cuda().detach(), reduction='none').sum()/args.batch_size
            loss_out = loss_out + loss_trans + loss_cnn + similarLoss/10000#
        if args.model == 's2net':
            k = 3
            alpha = 10
            loss_spatial = F.binary_cross_entropy_with_logits(spat_logits,labels.cuda().detach(),reduction='none')
            loss_spectral = F.binary_cross_entropy_with_logits(spec_logits, labels.cuda().detach(),reduction='none')
            loss_overrall = F.binary_cross_entropy_with_logits(overall_logits, labels.cuda().detach(),reduction='none')
            loss_ensemble = F.binary_cross_entropy_with_logits(ensemble_logits, labels.cuda().detach(),reduction='none')

            vw = torch.softmax(model.vote_weight, dim=0).detach() # 梯度停止回传
            # vw = torch.ones_like(model.vote_weight,requires_grad=False)/3
            punish_weight = (alpha * torch.ones([k,args.num_labels])).cuda().pow(-torch.log(k*vw))
            # print(k*vw)
            # loss_spatial = (punish_weight[0] * loss_spatial).sum() / args.batch_size
            # loss_spectral = (punish_weight[1] * loss_spectral).sum() / args.batch_size
            # loss_overrall = (punish_weight[2] * loss_overrall).sum() / args.batch_size
            loss_spatial = (loss_spatial).sum() / args.batch_size
            loss_spectral = (loss_spectral).sum() / args.batch_size
            loss_overrall = (loss_overrall).sum() / args.batch_size
            loss_ensemble = loss_ensemble.sum() / args.batch_size


            # self_corr_matrix = torch.matmul(label_emb,label_emb.transpose(0,1))
            # unit_matrix = torch.eye(args.num_labels, requires_grad=False).cuda()
            # self_corr_loss = (self_corr_matrix - unit_matrix) * torch.abs(1-unit_matrix)
            # self_corr_loss = (self_corr_loss**2).mean()/100
            self_corr_matrix = torch.matmul(label_emb,label_emb.transpose(0,1)) # 向量间的内积 [17, 17]
            unit_matrix = torch.eye(args.num_labels, requires_grad=False).cuda() # 单位矩阵 [17, 17]
            module = ((label_emb**2).sum(dim=1)**0.5).reshape(-1,1) #各向量的模长 [17, 1]
            dividen = torch.matmul(module,module.transpose(0,1))+(1e-6)#各模长的乘积 [17, 17]
            self_corr_loss = torch.abs(self_corr_matrix/dividen - unit_matrix).sum() # L1 损失
            ##if self_corr_loss.item() < 0.005:
            ## print(self_corr_matrix)
            ##print(self_corr_matrix/dividen)
            pred = ensemble_logits

            # /epoch
            loss_out = loss_ensemble + loss_spatial + loss_overrall + loss_spectral + self_corr_loss/10 # loss_ensemble + loss_spatial +
        if args.model == 'tsformer':
            loss_out = F.binary_cross_entropy_with_logits(pred.view(labels.size(0),-1),labels.cuda().detach(),reduction='mean')
        if args.model == 'ida':
            loss_out = F.binary_cross_entropy_with_logits(pred.view(labels.size(0), -1), labels.cuda().detach(),
                                                          reduction='mean')


        # loss_out = loss_out/unk_mask.cuda().sum()
        if(args.model=='s2net'):
            pbar.set_description("%s - spat: %.2f  spec: %.2f  overall:%.2f ensemble:%.2f self_corr:%.2f"
                                 %(desc, loss_spatial.item(), loss_spectral.item(), loss_overrall.item(),loss_ensemble.item(), self_corr_loss.item()))
        else:
            pbar.set_description("%s - loss: %.2f"
                                 % (desc, loss_out.item()))

        if train:
            loss_out.backward()
            # Grad Accumulation
            if ((batch_idx+1)%args.grad_ac_steps == 0):
                optimizer.step()
                optimizer.zero_grad()
                if warmup_scheduler is not None:
                    warmup_scheduler.step()

        ## Updates ##
        loss_total += loss_out.item()
        unk_loss_total += loss_out.item()
        start_idx,end_idx=(batch_idx*data.batch_size),((batch_idx+1)*data.batch_size)
        
        if pred.size(0) != all_predictions[start_idx:end_idx].size(0):
            pred = pred.view(labels.size(0),-1)

        #print(pred.shape,end_idx-start_idx)
        all_predictions[start_idx:end_idx] = pred.data.cpu()
        all_targets[start_idx:end_idx] = labels.data.cpu()
        all_masks[start_idx:end_idx] = mask.data.cpu()
        batch_idx += 1

    if 's2net' in args.model:
        print(vw)

    loss_total = loss_total/float(all_predictions.size(0))
    unk_loss_total = unk_loss_total/float(all_predictions.size(0))
    if ('split_16c' in args.model) or ('mc16' in args.model) or ('together' in args.model):
        return all_predictions,all_targets,all_masks,all_image_ids,loss_total,unk_loss_total,attns
    else:
        return all_predictions,all_targets,all_masks,all_image_ids,loss_total,unk_loss_total,None


