#!/bin/env python 
# -*- coding: utf-8 -*-
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.model_ProGAN as module_arch_ProGAN
import model.model_UNCGAN as module_arch_UNCGAN
from parse_config import ConfigParser
from trainer import Trainer, Trainer_WGAN, Trainer_ProGAN, Trainer_UNCGAN, Trainer_UNCGAN_M
from utils import prepare_device, initialize_weights


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed()

def main(config):
    logger = config.get_logger('train')

    # Set Model Name
    model_name = config['trainer']['model_name']
    print('Model Name : ', model_name)

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    print('data loader : ',data_loader)
    # valid_data_loader = data_loader.split_validation()
    # print('valid data loader : ',valid_data_loader)
    
    print('prepare gpu training')
    device, device_ids = prepare_device(config['n_gpu'])

    # build model architecture, then print to console
    if model_name in ['UNCGAN', 'UNCTransGAN']:
        M = config['trainer']['M']
        if M==0:
            model_G = config.init_obj('arch_G', module_arch_UNCGAN)
            model_D = config.init_obj('arch_D', module_arch_UNCGAN)
        if M==1:
            model_G = config.init_obj('arch_G', module_arch_UNCGAN)
            model_D = config.init_obj('arch_D', module_arch_UNCGAN)

            ### Import The previously trained Generator
            model_G0 = config.init_obj('arch_G0', module_arch_UNCGAN)
            checkpoint_G0 = torch.load(config['trainer']['checkpoint_G0'])
            state_dict_G0 = checkpoint_G0['state_dict_G']
            if config['n_gpu'] > 1:
                model_G0 = torch.nn.DataParallel(model_G0,device_ids=device_ids)
            model_G0.load_state_dict(state_dict_G0)
            model_G0 = model_G0.to(device)


    if model_name in ['ProGAN']:
        model_G = config.init_obj('arch_G', module_arch_ProGAN)
        model_D = config.init_obj('arch_D', module_arch_ProGAN)  
    
    if model_name in ['DCGAN', 'HDCGAN', 'WGAN' ,'WGANGP']:
        model_G = config.init_obj('arch_G', module_arch)
        model_D = config.init_obj('arch_D', module_arch)
        
    print('model_G : ', model_G)
    print('model_D : ', model_D)
    logger.info(model_G)
    logger.info(model_D)

    # prepare for (multi-device) GPU training

    model_G = model_G.to(device)
    model_D = model_D.to(device)
    
    # Weight initialization
    model_G.apply(initialize_weights)
    model_D.apply(initialize_weights)


    if len(device_ids) > 1:
        model_G = torch.nn.DataParallel(model_G, device_ids=device_ids)
        model_D = torch.nn.DataParallel(model_D, device_ids=device_ids)

    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])

    
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    print('metrics : ',metrics)
    print('Model G parameters')
    print(model_G.parameters())
    print('Model D parameters')
    print(model_D.parameters())



    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params_G = filter(lambda p: p.requires_grad, model_G.parameters())
    trainable_params_D = filter(lambda p: p.requires_grad, model_D.parameters())
    print('trainable_params G : ',trainable_params_G)
    print('trainable_params D : ',trainable_params_D)
    optimizer_G = config.init_obj('optimizer_G', torch.optim, trainable_params_G)
    optimizer_D = config.init_obj('optimizer_D', torch.optim, trainable_params_D)
    

    # lr_scheduler_G = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer_G)
    # lr_scheduler_D = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer_D)

    lr_scheduler = None


    if model_name in ['UNCGAN','UNCTransGAN']:
        num_epochs = config['trainer']['epochs']
        lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, num_epochs)
        lr_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, num_epochs)
        
        criterion = None
        if M==0:
            trainer = Trainer_UNCGAN(model_G,model_D, criterion, metrics, optimizer_G,optimizer_D,
                        config=config,
                        device=device,
                        data_loader=data_loader,
                        valid_data_loader=None,
                        lr_scheduler_G=lr_scheduler_G,
                        lr_scheduler_D=lr_scheduler_D)
        
        if M==1:
            trainer = Trainer_UNCGAN_M(model_G,model_G0, model_D, criterion, metrics, optimizer_G,optimizer_D,
                        config=config,
                        device=device,
                        data_loader=data_loader,
                        valid_data_loader=None,
                        lr_scheduler_G=lr_scheduler_G,
                        lr_scheduler_D=lr_scheduler_D)
        trainer.train_UNCGAN()
    

    if model_name in ['DCGAN', "HDCGAN"]:
        criterion = torch.nn.BCELoss()
        trainer = Trainer(model_G,model_D, criterion, metrics, optimizer_G,optimizer_D,
                        config=config,
                        device=device,
                        data_loader=data_loader,
                        valid_data_loader=None,
                        lr_scheduler=lr_scheduler)
        trainer.train()
    
    if model_name in ['WGAN', 'WGANGP']:
        criterion = None
        trainer = Trainer_WGAN(model_G,model_D, criterion, metrics, optimizer_G,optimizer_D,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=None,
                      lr_scheduler=lr_scheduler)
        trainer.train()

    if model_name in ['ProGAN']:
        criterion = None
        trainer = Trainer_ProGAN(model_G,model_D, criterion, metrics, optimizer_G,optimizer_D,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=None,
                      lr_scheduler=lr_scheduler)
        trainer.train_ProGAN()


    
    # trainer.train()


if __name__ == '__main__':
    # ArgumentParser 객체 생성
    args = argparse.ArgumentParser(description='PyTorch Template')
    
    # 인자 추가하기 - 명령행의 문자열을 객체로 변환하는 방법을 알려줌
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r_G', '--resume_G', default=None, type=str,
                      help='path to latest checkpoint (GENERATOR) (default: None)')
    args.add_argument('-r_D', '--resume_D', default=None, type=str,
                      help='path to latest checkpoint (DISCRIMINATOR) (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    # custom cli options to modify configuration from default values given in json file.
    # config.JSON 에서 주어진 옵션을 사용하고, 변경할 것 있으면 이것을 이용하여 변경
    # ex. python train.py -c config.json --bs 256
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--opt_G', '--G_LR'], type=float, target='optimizer_G;args;lr'),
        CustomArgs(['--opt_D', '--D_LR'], type=float, target='optimizer_D;args;lr'),
        CustomArgs(['--ngf', '--ngf_num'], type=int, target='arch_G;args;ngf'),
        CustomArgs(['--ndf', '--ndf_num'], type=int, target='arch_D;args;ndf'),
        CustomArgs(['--g_kernel', '--G_kernel_size'], type=int, target='arch_G;args;G_kernel_size'),
        CustomArgs(['--d_kernel', '--D_kernel_size'], type=int, target='arch_D;args;D_kernel_size'),
        CustomArgs(['--nz_model', '--nz_in_model'], type=int, target='arch_G;args;nz'),
        CustomArgs(['--nz_trainer', '--nz_in trainer'], type=int, target='trainer;nz')
    ]
    config = ConfigParser.from_args(args, options)
    print('begin main')
    main(config)
