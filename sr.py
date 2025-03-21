import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import torch.nn as nn
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import open_clip
import time
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/home/xys/IJCAI2024/SR3-CLIP/config/SR3-CLIP2.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if torch.backends.cudnn.is_available():
        print("cuDNN is available.")
    else:
        print("cuDNN is not available.")
        
    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')
#------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLIP
    # clip_model, _preprocess = clip.load("ViT-B/32", device=device)
    if opt['model']['daclip']['dalcip_path'] is not None:
        clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=opt['model']['daclip']['dalcip_path'])
    else:
        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    # 使用DataParallel包装模型
    clip_model = nn.DataParallel(clip_model)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    degradation = opt['distortion']
    degradation_text = tokenizer(degradation)
    degradation_text = degradation_text.to("cuda")
    clip_model = clip_model.to("cuda")
#------------------------------------------------------------------------------------------------------------------------------------------------------

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                # 模糊程度判断
                with torch.no_grad(), torch.cuda.amp.autocast():
                    

                    degradation_text_features = clip_model.module.encode_text(degradation_text)
                    VI_image_context, VI_degra_context = clip_model.module.encode_image(train_data['clip_img_VI_SR'], control=True)
                    IR_image_context, IR_degra_context = clip_model.module.encode_image(train_data['clip_img_IR_SR'], control=True)
                    norm_VI_degra_context = VI_degra_context / VI_degra_context.norm(dim=-1, keepdim=True)
                    norm_IR_degra_context = IR_degra_context / IR_degra_context.norm(dim=-1, keepdim=True)
                    degradation_text_features /= degradation_text_features.norm(dim=-1, keepdim=True)
                    # VI_degra_context /= VI_degra_context.norm(dim=-1, keepdim=True)
                    # IR_degra_context /= IR_degra_context.norm(dim=-1, keepdim=True)

                    VI_text_probs = (100.0 * norm_VI_degra_context @ degradation_text_features.T).softmax(dim=-1)
                    IR_text_probs = (100.0 * norm_IR_degra_context @ degradation_text_features.T).softmax(dim=-1)
                    VI_index = torch.argmax(VI_text_probs[0])
                    IR_index = torch.argmax(IR_text_probs[0])
                    if degradation[VI_index] == 'blur' and degradation[IR_index] == 'Clarity':
                        image_context = VI_image_context.float()
                        degra_context = VI_degra_context.float()
                    elif degradation[IR_index] == 'blur' and degradation[VI_index] == 'Clarity':
                        image_context = IR_image_context.float()
                        degra_context = IR_degra_context.float()
                    else:
                        if torch.all(train_data['text'].eq(0))==False:
                            
                            # text = tokenizer(train_data['text'])
                            _, degra_context = clip_model.module.encode_image(train_data['clip_img_VI_SR'], control=True)
                            image_context = train_data['text']
                        else:
                            VI_image_context, VI_degra_context = clip_model.module.encode_image(train_data['clip_img_VI_SR'], control=True)
                            IR_image_context, IR_degra_context = clip_model.module.encode_image(train_data['clip_img_IR_SR'], control=True)
                            # image_context = (VI_image_context+IR_image_context)/2
                            image_context = torch.maximum(VI_image_context, IR_image_context)
                            
                            # degra_context = (VI_degra_context+IR_degra_context)/2
                            degra_context = torch.maximum(VI_degra_context, IR_degra_context)
                
                train_data['image_context'] = image_context
                train_data['degra_context'] = degra_context

                
                # diffuson_start_time = time.time()

                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()

                # diffuson_end_time = time.time()
                # print(f"Diffusion计算使用时长: {diffuson_end_time - diffuson_start_time} seconds")

                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        with torch.no_grad(), torch.cuda.amp.autocast():
                            degradation_text_features = clip_model.module.encode_text(degradation_text)
                            VI_image_context, VI_degra_context = clip_model.module.encode_image(val_data['clip_img_VI_SR'], control=True)
                            IR_image_context, IR_degra_context = clip_model.module.encode_image(val_data['clip_img_IR_SR'], control=True)
                            norm_VI_degra_context = VI_degra_context / VI_degra_context.norm(dim=-1, keepdim=True)
                            norm_IR_degra_context = IR_degra_context / IR_degra_context.norm(dim=-1, keepdim=True)
                            degradation_text_features /= degradation_text_features.norm(dim=-1, keepdim=True)
                            # VI_degra_context /= VI_degra_context.norm(dim=-1, keepdim=True)
                            # IR_degra_context /= IR_degra_context.norm(dim=-1, keepdim=True)

                            VI_text_probs = (100.0 * norm_VI_degra_context @ degradation_text_features.T).softmax(dim=-1)
                            IR_text_probs = (100.0 * norm_IR_degra_context @ degradation_text_features.T).softmax(dim=-1)
                            VI_index = torch.argmax(VI_text_probs[0])
                            IR_index = torch.argmax(IR_text_probs[0])
                            if degradation[VI_index] == 'blur' and degradation[IR_index] == 'Clarity':
                                image_context = VI_image_context.float()
                                degra_context = VI_degra_context.float()
                            elif degradation[IR_index] == 'blur' and degradation[VI_index] == 'Clarity':
                                image_context = IR_image_context.float()
                                degra_context = IR_degra_context.float()
                            else:
                                if torch.all(val_data['text'].eq(0))==False:
                                    # text = tokenizer(val_data['text'])
                                    # text_features = clip_model.encode_text(text)
                                    _, degra_context = clip_model.module.encode_image(val_data['VI'], control=True)
                                    image_context = val_data['text']
                                else:
                                    VI_image_context, VI_degra_context = clip_model.module.encode_image(val_data['clip_img_VI_SR'], control=True)
                                    IR_image_context, IR_degra_context = clip_model.module.encode_image(val_data['clip_img_IR_SR'], control=True)
                                    image_context = (VI_image_context+IR_image_context)/2
                                    degra_context = (VI_degra_context+IR_degra_context)/2
                            val_data['image_context'] = image_context
                            val_data['degra_context'] = degra_context

                            diffusion.feed_data(val_data)
                            diffusion.test(continous=False)
                            visuals = diffusion.get_current_visuals()
                            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                            IR_lr_img = Metrics.tensor2img(visuals['IR_INF'])  # uint8
                            VI_lr_img = Metrics.tensor2img(visuals['VI_INF'])  # uint8
                            fake_img = Metrics.tensor2img(visuals['VI_INF'])  # uint8

                            # generation
                            Metrics.save_img(
                                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
                                sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
                                IR_lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
                                VI_lr_img, '{}/{}_{}_VI.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
                                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                            tb_logger.add_image(
                                'Iter_{}'.format(current_step),
                                np.transpose(np.concatenate(
                                    (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                                idx)
                            avg_psnr += Metrics.calculate_psnr(
                                sr_img, hr_img)

                            if wandb_logger:
                                wandb_logger.log_image(
                                    f'validation_{idx}', 
                                    np.concatenate((fake_img, sr_img, hr_img), axis=1)
                                )

                        avg_psnr = avg_psnr / idx
                        diffusion.set_new_noise_schedule(
                            opt['model']['beta_schedule']['train'], schedule_phase='train')
                        # log
                        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                        logger_val = logging.getLogger('val')  # validation logger
                        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                            current_epoch, current_step, avg_psnr))
                        # tensorboard logger
                        tb_logger.add_scalar('psnr', avg_psnr, current_step)

                        if wandb_logger:
                            wandb_logger.log_metrics({
                                'validation/val_psnr': avg_psnr,
                                'validation/val_step': val_step
                            })
                            val_step += 1

                    if current_step % opt['train']['save_checkpoint_freq'] == 0:
                        logger.info('Saving models and training states.')
                        diffusion.save_network(current_epoch, current_step)

                        if wandb_logger and opt['log_wandb_ckpt']:
                            wandb_logger.log_checkpoint(current_epoch, current_step)

                if wandb_logger:
                    wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
