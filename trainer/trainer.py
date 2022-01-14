import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, get_infinite_batches, gradient_penalty, gradient_penalty_ProGAN
from data_loader.data_loaders import KNUskinDataLoader_ProGAN


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss_G','loss_D', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model_G.train()
        self.model_D.train()

        self.train_metrics.reset()
        for batch_idx, (data,_) in enumerate(self.data_loader):

            data = data.to(self.device)

            valid = torch.full((data.shape[0],1,1,1),1).view(-1,1,1,1).to(self.device).float()
            fake = torch.full((data.shape[0],1,1,1),0).view(-1,1,1,1).to(self.device).float()

            z = torch.normal(0, 1, size=(data.shape[0], self.nz,1,1)).to(self.device)

            # ----------------------------------
            #  Train Generator
            # ----------------------------------
            self.optimizer_G.zero_grad()

            # Feed Forward
            gen_imgs = self.model_G(z)
            # Loss Measures generators ability to fool the discriminator
            loss_G = self.criterion(self.model_D(gen_imgs), valid)
            # Backpropagation
            loss_G.backward()
            self.optimizer_G.step()

            # --------------------------------------
            #  Train Discriminator
            # --------------------------------------
            self.optimizer_D.zero_grad()

            # Feed forward
            y_real = self.model_D(data)
            y_fake = self.model_D(gen_imgs.detach())
            # Measure discriminator's ability to classify real from generated samples
            real_loss = self.criterion(y_real,valid)
            fake_loss = self.criterion(y_fake,fake)
            loss_D = (real_loss+fake_loss)/2
            # Backpropagation
            loss_D.backward()
            self.optimizer_D.step()


            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss_G', loss_G.item())
            self.train_metrics.update('loss_D', loss_D.item())
            
            # TRACK loss_D (REAL), loss_D (Fake)
            self.writer.add_scalar('loss_D_REAL', real_loss)
            self.writer.add_scalar('loss_D_FAKE', fake_loss)
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss_G: {:.4f}, Loss_D: {:.4f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_G.item(),
                    loss_D.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation: #NOT RUN
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None: #None임
            self.lr_scheduler.step()
        # print('gen_imgs shape: ', gen_imgs.shape)
        # print('gen_imgs.data[:9] shape: ', gen_imgs.data[:9].shape)
        
        if epoch==0 or epoch%5==0:
            self.writer.add_image('Generated_Image', make_grid(gen_imgs.data[:6].cpu(), nrow=3, normalize=True))

        return log


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)




class Trainer_WGAN(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.type_name = self.config['name']

        self.critic_iter = self.config['trainer']['critic_iter']

        if 'WGANGP' == self.model_name:                                  ## WGAN gradient penalty
            self.lambda_GP = self.config['trainer']['lambda_GP']
        if 'WGAN' == self.model_name:                                    ## WGAN clipping
            self.weight_cliping_limit = self.config['trainer']['weight_cliping_limit']


        self.train_metrics = MetricTracker('loss_G','loss_D', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model_G.train()
        self.model_D.train()

        self.train_metrics.reset()

        for batch_idx, (data,_) in enumerate(self.data_loader):

            real = data.to(self.device)

            if 'WGANGP' in self.type_name: 

                # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
                for d_iter in range(self.critic_iter):

                    z = torch.normal(0, 1, size=(real.shape[0], self.nz,1,1)).to(self.device)

                    # Train Discriminator - WGAN - Training discriminator more iterations than generator
                    # Train with real images
                    fake = self.model_G(z)
                    d_loss_real = self.model_D(real).reshape(-1)
                    d_loss_fake = self.model_D(fake).reshape(-1)
                    gp = gradient_penalty(self.model_D,real,fake,device=self.device)
                    loss_D = (
                        -(torch.mean(d_loss_real)-torch.mean(d_loss_fake)) + self.lambda_GP*gp
                    )

                    self.model_D.zero_grad()
                    loss_D.backward(retain_graph=True)      # Reutilize the computations for 'fake' when we do the updates for the generator
                    self.optimizer_D.step()

                ### Train Generator: min -E[D(gen_fake)]
                output = self.model_D(fake).reshape(-1)
                loss_G = -torch.mean(output)
                self.model_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()

            else: # if model==WGAN
                # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
                for d_iter in range(self.critic_iter):

                    z = torch.normal(0, 1, size=(real.shape[0], self.nz,1,1)).to(self.device)

                    # Train Discriminator - WGAN - Training discriminator more iterations than generator
                    # Train with real images
                    fake = self.model_G(z)
                    d_loss_real = self.model_D(real).reshape(-1)
                    d_loss_fake = self.model_D(fake).reshape(-1)
                    loss_D = -(torch.mean(d_loss_real)-torch.mean(d_loss_fake))

                    self.model_D.zero_grad()
                    loss_D.backward(retain_graph=True)      # Reutilize the computations for 'fake' when we do the updates for the generator
                    self.optimizer_D.step()

                    # Clamp parameters to a range [-c, c], c=self.weight_cliping_limit
                    for p in self.model_D.parameters():
                        p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                ### Train Generator: min -E[D(gen_fake)]
                output = self.model_D(fake).reshape(-1)
                loss_G = -torch.mean(output)
                self.model_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()
                
                

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss_G', loss_G.item())
            self.train_metrics.update('loss_D', loss_D.item())
            
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss_G: {:.4f}, Loss_D: {:.4f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_G.item(),
                    loss_D.item())),
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            # if batch_idx == self.len_epoch:
            #     break
        log = self.train_metrics.result()

        if self.do_validation: #NOT RUN
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None: #None임
            self.lr_scheduler.step()

        if epoch==0 or epoch%5==0:
            self.writer.add_image('Generated_Image', make_grid(fake.data[:6].cpu(), nrow=3, normalize=True))

        return log


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)



#########################################################################################################
### Trainer_ProGAN
#########################################################################################################



class Trainer_ProGAN(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model_G, model_D, criterion, metric_ftns, optimizer_G, optimizer_D, config)
        self.config = config
        self.device = device
        self.data_loader_dummy = data_loader
        
        # if len_epoch is None:
        #     # epoch-based training
        #     self.len_epoch = len(self.data_loader_dummy)
        # else:
        #     # iteration-based training
        #     self.data_loader = inf_loop(data_loader)
        #     self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.type_name = self.config['name']

        # Set Gradient Penalty Parameter
        self.lambda_GP = self.config['trainer']['lambda_GP']


        dlp = config['data_loader']['args']        # for calling dataloader parameters
        self.dlp_data_dir = dlp['data_dir']
        self.dlp_subject_map = dlp['subject_map']
        self.dlp_cell_type = dlp['cell_type']
        self.dlp_shuffle = dlp['shuffle']
        self.dlp_validation_split = dlp['validation_split']
        self.dlp_num_workers = dlp['num_workers']

        self.fixed_noise = torch.normal(0, 1, size=(16, self.nz,1,1)).to(self.device)


        self.train_metrics = MetricTracker('loss_G','loss_D', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model_G.train()
        self.model_D.train()

        self.train_metrics.reset()

        if self.use_autocast:
            scaler_D = torch.cuda.amp.GradScaler()
            scaler_G = torch.cuda.amp.GradScaler()

        ### GET DATALOADER according to img size, batch size
        self.data_loader = KNUskinDataLoader_ProGAN(
            data_dir=self.dlp_data_dir,
            batch_size=self.batch_size,
            input_size=self.img_size,
            subject_map=self.dlp_subject_map,
            cell_type=self.dlp_cell_type,
            shuffle=self.dlp_shuffle,
            validation_split=self.dlp_validation_split,
            num_workers=self.dlp_num_workers)
        
        
        self.len_epoch = len(self.data_loader)

        for batch_idx, (data,_) in enumerate(self.data_loader):
            # print('alpha : ', self.alpha)
            real = data.to(self.device)

            z = torch.normal(0, 1, size=(real.shape[0], self.nz,1,1)).to(self.device)

            if self.use_autocast:
                with torch.cuda.amp.autocast():
                # Train Discriminator : MAX (E[critic(real)]-E[critic(fake)])
                    fake = self.model_G(z, self.alpha, self.step)
                    d_loss_real = self.model_D(real,self.alpha,self.step).reshape(-1)
                    d_loss_fake = self.model_D(fake.detach(),self.alpha, self.step)
                    gp = gradient_penalty_ProGAN(self.model_D, real, fake, self.alpha, self.step, device=self.device)
                    loss_D = (
                        -(torch.mean(d_loss_real)-torch.mean(d_loss_fake))
                        + self.lambda_GP*gp
                        + (0.001 * torch.mean(d_loss_real**2))
                    )

                self.optimizer_D.zero_grad()
                scaler_D.scale(loss_D).backward(retain_graph=True)
                scaler_D.step(self.optimizer_D)
                scaler_D.update()


                # Train Generator: MAX E[D(gen_fake)]  <=> MIN -E[D(gen_fake)]
                with torch.cuda.amp.autocast():
                    output = self.model_D(fake,self.alpha,self.step)
                    loss_G = -torch.mean(output)

                self.optimizer_G.zero_grad()
                scaler_G.scale(loss_G).backward()
                scaler_G.step(self.optimizer_G)
                scaler_G.update()
            
            else: ## Not use autocast
                
                # Train Discriminator : MAX (E[critic(real)]-E[critic(fake)])
                fake = self.model_G(z, self.alpha, self.step)
                d_loss_real = self.model_D(real,self.alpha,self.step).reshape(-1)
                d_loss_fake = self.model_D(fake.detach(),self.alpha, self.step)
                gp = gradient_penalty_ProGAN(self.model_D, real, fake, self.alpha, self.step, device=self.device)
                loss_D = (
                    -(torch.mean(d_loss_real)-torch.mean(d_loss_fake))
                    + self.lambda_GP*gp
                    + (0.001 * torch.mean(d_loss_real**2))
                )
                
                self.model_D.zero_grad()
                loss_D.backward(retain_graph=True)
                self.optimizer_D.step()
                
                # Train Generator: MAX E[D(gen_fake)]  <=> MIN -E[D(gen_fake)]
                output = self.model_D(fake,self.alpha,self.step)
                loss_G = -torch.mean(output)
                
                self.model_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()
                
                
            # print('realshape : ', real.shape[0])
            # print('len(self.data_loader.dataset) : ', len(self.data_loader.dataset))
            self.alpha += real.shape[0]/ (len(self.data_loader.dataset)*self.progressive_epochs[self.step]*0.5)
            self.alpha = min(self.alpha,1)
            # print('self.alpha += real.shape[0]/ len(self.data_loader.dataset) = ',self.alpha)
            

            self.writer.set_step(self.tensor_step)
            self.train_metrics.update('loss_G', loss_G.item())
            self.train_metrics.update('loss_D', loss_D.item())
            
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch {} (ImgSize: {}) {} Loss_G: {:.4f}, Loss_D: {:.4f}'.format(
                    epoch,
                    self.img_size,
                    self._progress(batch_idx),
                    loss_G.item(),
                    loss_D.item())),
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            # if batch_idx == self.len_epoch:
            #     break
            
            self.tensor_step+=1

        log = self.train_metrics.result()

        if self.do_validation: #NOT RUN
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None: #None임
            self.lr_scheduler.step()

        if epoch==1 or epoch%5==0:
            with torch.no_grad():
                random_fake = self.model_G(z, self.alpha, self.step)
                fixed_fake = self.model_G(self.fixed_noise, self.alpha, self.step)
                # Generated Random Image
                self.writer.add_image('Generated_Images', make_grid(random_fake.data[:16].cpu(), nrow=4, normalize=True))
                # Generated Fixed Image
                self.writer.add_image('Generated_Images (Fixed)', make_grid(fixed_fake.data[:16].cpu(), nrow=4, normalize=True))
                if epoch==1:
                    # Imput Image
                    self.writer.add_image('Input_Images', make_grid(data[:16].cpu(), nrow=4, normalize=True))

        return log


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

