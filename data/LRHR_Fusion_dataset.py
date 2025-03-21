from io import BytesIO
import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import data.util as Util
import pandas as pd
import open_clip
class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, csv_file_path,clip_path, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.csv_file_path = csv_file_path
        self.clip_path = clip_path
        # load CLIP model
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model, self.preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32',pretrained=self.clip_path)
        
        if datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        
        elif datatype == 'Fusion':
            self.ir_sr_path = Util.get_paths_from_images(
                '{}/ir_sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.vi_sr_path = Util.get_paths_from_images(
                '{}/vi_sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR: # 训练时不需要低分辨率图像
                self.ir_lr_path = Util.get_paths_from_images(
                    '{}/ir_lr_{}'.format(dataroot, l_resolution))
                self.vi_lr_path = Util.get_paths_from_images(
                    '{}/vi_lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
            
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_Fusion_HR = None
        img_VI_LR = None
        img_IR_LR = None
        img_VI_SR = None
        img_IR_SR = None
        # 读取csv文件
        

        csv_file = pd.read_csv(self.csv_file_path)
        token_text = torch.zeros((1,77))
        token_text = token_text.long()
        encode_text = torch.zeros((1,512))
        
        if self.datatype == 'Fusion':
            img_Fusion_HR = Image.open(self.hr_path[index]).convert("RGB") # 高分辨率
            img_VI_SR = Image.open(self.vi_sr_path[index]).convert("RGB")
            img_IR_SR = Image.open(self.ir_sr_path[index]).convert("RGB")
            text = Util.get_image_description(csv_file, self.hr_path[index])
            if text is not None:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    token_text = self.tokenizer(text)
                    encode_text = self.model.encode_text(token_text)
            
            if self.need_LR:
                img_VI_LR = Image.open(self.vi_lr_path[index]).convert("RGB")
                img_IR_LR = Image.open(self.ir_lr_path[index]).convert("RGB")
        if self.need_LR:
            [clip_img_VI_SR, clip_img_IR_SR] = Util.transform_augment( 
                [img_VI_SR, img_IR_SR], split=self.split, min_max=(-1, 1), size=(224,224))
            
            [img_VI_LR, img_IR_LR, img_VI_SR,img_IR_SR, img_Fusion_HR] = Util.transform_augment( 
                [img_VI_LR, img_IR_LR, img_VI_SR, img_IR_SR, img_Fusion_HR], split=self.split, min_max=(-1, 1))
            
            
            return {'VI_LR': img_VI_LR, 'IR_LR': img_IR_LR, 'VI': img_VI_SR, 'IR': img_IR_SR, 'HR': img_Fusion_HR, 'Index': index, 
                    'VI_IR_LR': torch.cat([img_IR_LR, img_VI_LR], dim=0),
                     'VI_IR_SR': torch.cat([img_IR_SR, img_VI_SR], dim=0), 'text': encode_text, 'clip_img_VI_SR': clip_img_VI_SR, 'clip_img_IR_SR': clip_img_IR_SR
                     }
        else:
            [clip_img_VI_SR, clip_img_IR_SR] = Util.transform_augment( 
                [img_VI_SR, img_IR_SR], split=self.split, min_max=(-1, 1), size=(224,224))
            
            [img_VI_SR, img_IR_SR, img_Fusion_HR] = Util.transform_augment(
                [img_VI_SR, img_IR_SR, img_Fusion_HR], split=self.split, min_max=(-1, 1))
            # [img_SR, img_Fusion_HR] = Util.transform_augment(
            #     [img_SR, img_Fusion_HR], split=self.split, min_max=(-1, 1))
            
            # print(img_VI_SR.shape, img_IR_SR.shape, img_Fusion_HR.shape)
            return {'VI': img_VI_SR, 'IR': img_IR_SR, 'HR': img_Fusion_HR, 'Index': index, 
                    'VI_IR_SR': torch.cat([img_IR_SR, img_VI_SR], dim=0),'text': encode_text, 'clip_img_VI_SR': clip_img_VI_SR, 'clip_img_IR_SR': clip_img_IR_SR}
            # return {'HR': img_Fusion_HR, 'SR': img_SR, 'Index': index}
