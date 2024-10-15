import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import open_clip
import torch.nn as nn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/public/home/xys/IJCAI2024/Mamba-SR3-CLIP/config/Mamba-SR3-CLIP2.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

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


    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
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
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals(need_LR=False)

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            fake_VI_img = Metrics.tensor2img(visuals['VI_INF'])  # uint8
            fake_IR_img = Metrics.tensor2img(visuals['IR_INF'])  # uint8
            # fake_img = Metrics.tensor2img(visuals['SR'])  # uint8
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
                fake_VI_img, '{}/{}_{}_VI_inf.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_IR_img, '{}/{}_{}_IR_inf.png'.format(result_path, current_step, idx))

            if wandb_logger and opt['log_infer']:
                wandb_logger.log_eval_data(fake_VI_img, fake_IR_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img)

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_table(commit=True)
