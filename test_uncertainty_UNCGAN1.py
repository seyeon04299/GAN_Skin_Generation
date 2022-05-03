import argparse
import torch
from tqdm import tqdm
from datetime import datetime
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer, Trainer_WGAN, Trainer_ProGAN, Trainer_UNCGAN, Trainer_UNCGAN_M
import model.model_UNCGAN as module_arch_UNCGAN

from utils import inf_loop, MetricTracker, get_infinite_batches, gradient_penalty, gradient_penalty_ProGAN, calculate_fretchet, prepare_device
from data_loader.data_loaders import KNUskinDataLoader_ProGAN
from model import InceptionV3
import torch.nn.functional as F
from utils import bayeLq_loss,bayeGen_loss, bayeLq_loss1, bayeLq_loss_n_ch, Sinogram_loss, bayeLq_Sino_loss, bayeLq_Sino_loss1,sigma_calc



def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=4,
        subject_map="data/ISIC2017/ISIC_2017_subjectmap.csv",
        cell_type1 = 'NEV',
        cell_type2='MEL',
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2,
        
    )

    cfg_trainer = config['trainer']
    list_epochs = cfg_trainer['list_epochs']
    list_lambda1 = cfg_trainer['list_lambda1']
    list_lambda2 = cfg_trainer['list_lambda2']

    # build model architecture
    model_G = config.init_obj('arch_G', module_arch_UNCGAN)
    logger.info(model_G)
    model_D = config.init_obj('arch_D', module_arch_UNCGAN)
    logger.info(model_D)

    # get function handles of loss and metrics
    # loss_fn_G = getattr(module_loss, config['loss_G'])
    # loss_fn_D = getattr(module_loss, config['loss_D'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume_G))
    logger.info('Loading checkpoint: {} ...'.format(config.resume_D))
    checkpoint_G = torch.load(config.resume_G)
    state_dict_G = checkpoint_G['state_dict_G']
    if config['n_gpu'] > 1:
        model_G = torch.nn.DataParallel(model_G)
    model_G.load_state_dict(state_dict_G)

    checkpoint_D = torch.load(config.resume_D)
    state_dict_D = checkpoint_D['state_dict_D']
    if config['n_gpu'] > 1:
        model_D = torch.nn.DataParallel(model_D)
    model_D.load_state_dict(state_dict_D)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])
    model=model.to(device)

    model_G = model_G.to(device)
    model_D = model_D.to(device)
    # model_G = torch.nn.DataParallel(model_G, device_ids=[0,1])
    # model_D = torch.nn.DataParallel(model_D, device_ids=[0,1])
    model_G.eval()
    model_D.eval()

    total_loss_G = 0.0
    total_loss_D = 0.0
    mean_sigma_tmp=0
    avg_tot_loss_tmp=0
    loss_D_tmp=0
    fretchet_dist_tmp=0
    
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for lam1, lam2 in zip(list_lambda1, list_lambda2):
            for i, loaded_data in enumerate(tqdm(data_loader)):
    

                data1 = loaded_data[0].to(device)
                data2 = loaded_data[1]

                data1 = data1.to(device)
                data2 = data2.to(device)
                # print('data_loaded')
                rec_B, rec_alpha_B, rec_beta_B = model_G(data1)

                # print(rec_B.shape)              # (8,3,224,224)
                # print(rec_alpha_B.shape)        # (1,3,224,224)
                # print(rec_beta_B.shape)         # (1,3,224,224)

                #first gen
                # model_D.eval()
                total_loss = lam1*F.l1_loss(rec_B, data2) + lam2*bayeGen_loss(rec_B, rec_alpha_B, rec_beta_B, data2)
                t0 = model_D(rec_B)
                t1 = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                e5 = 0.001*F.mse_loss(t1, torch.ones(t1.size()).to(device).type(torch.cuda.FloatTensor))
                total_loss += e5
                # print('Total loss calculated')

                #then discriminator
                
                t0 = model_D(data2)
                pred_real_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                loss_D_A_real = 1*F.mse_loss(
                    pred_real_A, torch.ones(pred_real_A.size()).to(device).type(torch.cuda.FloatTensor)
                )
                t0 = model_D(rec_B.detach())
                pred_fake_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                loss_D_A_pred = 1*F.mse_loss(
                    pred_fake_A, torch.zeros(pred_fake_A.size()).to(device).type(torch.cuda.FloatTensor)
                )
                loss_D_A = (loss_D_A_real + loss_D_A_pred)*0.5


                loss_D = loss_D_A
                # print('loss_D calculated')

                avg_tot_loss_tmp += total_loss.item()
                loss_D_tmp += loss_D

                fretchet_dist=calculate_fretchet(data2,rec_B,model)
                fretchet_dist_tmp += fretchet_dist

                sigma_uncertainty = sigma_calc(rec_alpha_B,rec_beta_B)
                mean_sigma = torch.mean(sigma_uncertainty)
                mean_sigma_tmp += mean_sigma
                # print('mean_sigma calculated')


                # for i, metric in enumerate(metric_fns):
                #     total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)


    log = {
        'loss_D': loss_D_tmp / n_samples,
        'FID': fretchet_dist_tmp/n_samples,
        'avg_tot_loss':avg_tot_loss_tmp/n_samples,
        'avg_sigma':mean_sigma_tmp/n_samples
    }
    print(log)
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r_G', '--resume_G', default=None, type=str,
                      help='path to latest checkpoint Generator (default: None)')
    args.add_argument('-r_D', '--resume_D', default=None, type=str,
                      help='path to latest checkpoint Discriminator (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args)
    main(config)
