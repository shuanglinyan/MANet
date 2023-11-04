# -*- coding: utf-8 -*-

from option.options import options, config
from data.dataloader import get_dataloader
import torch
from model.model import TextImgPersonReidNet
from loss.Id_loss import Id_Loss
from loss.RankingLoss import RankingLoss
from loss.CRLoss import CRLoss
from solver import WarmupMultiStepLR
from torch import optim
import logging
import os
from test_during_train import test
from torch.autograd import Variable
import numpy as np
import random


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def save_checkpoint(state, opt):

    filename = os.path.join(opt.save_path, 'model/best.pth.tar')
    torch.save(state, filename)

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(opt):
    opt.device = torch.device('cuda:{}'.format(opt.GPU_id))
    fix_seed(opt.seed)
    opt.save_path = './checkpoints/{}/'.format(opt.dataset) + opt.model_name

    print('Num of centers: {}'.format(opt.num_clusters))

    config(opt)
    train_dataloader = get_dataloader(opt)
    opt.mode = 'test'
    test_img_dataloader, test_txt_dataloader = get_dataloader(opt)
    opt.mode = 'train'

    id_loss_fun_global = Id_Loss(opt, 1, 2048).to(opt.device)
    id_loss_fun_local = Id_Loss(opt, opt.num_clusters, 512).to(opt.device)
    ranking_loss_fun = RankingLoss(opt)
    cr_loss_fun = CRLoss(opt)
    network = TextImgPersonReidNet(opt).to(opt.device)

    cnn_params = list(map(id, network.ImageExtract.parameters()))
    other_params = filter(lambda p: id(p) not in cnn_params, network.parameters())
    other_params = list(other_params)
    other_params.extend(list(id_loss_fun_global.parameters()))
    other_params.extend(list(id_loss_fun_local.parameters()))
    param_groups = [{'params': other_params, 'lr': opt.lr},
                    {'params': network.ImageExtract.parameters(), 'lr': opt.lr * 0.1}]

    optimizer = optim.Adam(param_groups, betas=(opt.adam_alpha, opt.adam_beta))

    test_best = 0
    test_history = 0

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.epoch_decay)
    scheduler = WarmupMultiStepLR(optimizer, (30, 50), 0.1, 0.01, 10, 'linear')

    for epoch in range(opt.epoch):

        id_loss_sum = 0
        ranking_loss_sum = 0
        loss_sum = 0

        for param in optimizer.param_groups:
            logging.info('lr:{}'.format(param['lr']))

        for times, [image, label, caption, caption_length, caption_cr, caption_length_cr] in enumerate(
                train_dataloader):

            (
                tokens,
                segments,
                input_masks,
                _,
            ) = network.TextExtract.language_model.pre_process(caption)
            (
                tokens_cr,
                segments_cr,
                input_masks_cr,
                _,
            ) = network.TextExtract.language_model.pre_process(caption_cr)

            image = Variable(image.to(opt.device))
            label = Variable(label.to(opt.device))

            tokens = Variable(tokens.to(opt.device).long())
            segments = Variable(segments.to(opt.device).long())
            input_masks = Variable(input_masks.to(opt.device).long())

            tokens_cr = Variable(tokens_cr.to(opt.device).long())
            segments_cr = Variable(segments_cr.to(opt.device).long())
            input_masks_cr = Variable(input_masks_cr.to(opt.device).long())

            img_pre, img_aft, img_global, img_local, txt_global, txt_local = network(image, tokens, segments, input_masks)
            txt_global_cr, txt_local_cr = network.txt_embedding(tokens_cr, segments_cr, input_masks_cr)

            id_loss_global = id_loss_fun_global(img_global, txt_global, label)
            id_loss_local = id_loss_fun_local(img_local, txt_local, label)
            id_loss = id_loss_global + id_loss_local

            cr_loss_rec = ranking_loss_fun(img_pre, img_aft, label, epoch >= opt.epoch_begin, semi=False)
            cr_loss_global = cr_loss_fun(img_global, txt_global, txt_global_cr, label, epoch >= opt.epoch_begin, semi=True)
            cr_loss_local = cr_loss_fun(img_local, txt_local, txt_local_cr, label, epoch >= opt.epoch_begin, semi=True)
            ranking_loss = cr_loss_global + cr_loss_local + cr_loss_rec
            #  + cr_loss_rec

            optimizer.zero_grad()
            loss = id_loss + ranking_loss
            loss.backward()
            optimizer.step()

            if (times + 1) % 50 == 0:
                logging.info("Epoch: %d/%d, iteration: %d, loss: %.2f, ranking_loss: %.2f, id_loss: %.2f"
                             % (epoch + 1, opt.epoch, times + 1, loss, ranking_loss, id_loss))

            ranking_loss_sum += ranking_loss
            id_loss_sum += id_loss
            loss_sum += loss
        ranking_loss_avg = ranking_loss_sum / (times + 1)
        id_loss_avg = id_loss_sum / (times + 1)
        loss_avg = loss_sum / (times + 1)

        logging.info("Epoch: %d/%d, loss: %.2f, ranking_loss: %.2f, id_loss: %.2f"
                     % (epoch + 1, opt.epoch, loss_avg, ranking_loss_avg, id_loss_avg))

        logging.info(opt.model_name)
        network.eval()
        test_best = test(opt, epoch + 1, network, test_img_dataloader, test_txt_dataloader, test_best)
        network.train()

        if test_best > test_history:
            test_history = test_best
            test_epoch = epoch + 1
            state = {
                'network': network.cpu().state_dict(),
                'test_best': test_best,
                'epoch': epoch
            }
            # save_checkpoint(state, opt)
            network.to(opt.device)

        scheduler.step()

    logging.info('Training Done')
    logging.info('***************************')
    logging.info('Num of centers: {}'.format(opt.num_clusters))
    logging.info('Best Epoch: {}'.format(test_epoch))
    logging.info('Best Top1: {:.2%}'.format(test_history))


if __name__ == '__main__':
    opt = options().opt
    train(opt)
