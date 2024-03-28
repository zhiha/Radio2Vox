import sys 
sys.path.append("./models/radio2vox")
import torch
import numpy as np
from utils.utils import iou
from pytorch_lightning import LightningModule
from visualizer import html_grid
from encoder import Encoder
from decoder import Decoder
from refiner import Refiner
import os

class Csi_Env3d_Radio2Vox(LightningModule):
    def __init__(self, learning_rate=0.0001,test_voxel=False) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.refiner = Refiner()
        self.learning_rate = learning_rate
        self.test_voxel = test_voxel
        self.wi_scene_shape=(60,60,30)
        self.automatic_optimization = False
    
    def forward1(self,x):
        with torch.no_grad():
            indices = torch.randperm(x.size(2))
            x = x[:, :, indices, :]
        e_x = self.encoder(x)
        d_x = self.decoder(e_x)
        return d_x
    
    def forward2(self,x):
        r_x = self.refiner(x)
        return r_x
    
    def training_step(self, batch, batch_idx):  
        opt1, opt2  = self.optimizers()
        csi_data_img, voxel_data = batch
        bce_loss = torch.nn.BCELoss()
        
        d_x = self.forward1(csi_data_img)
        d_x_sq = d_x.squeeze(1)
        iou1 = iou(d_x_sq.round().bool(), voxel_data.round().bool())
        loss1 = bce_loss(d_x_sq, voxel_data) * 10
        opt1.zero_grad()
        self.manual_backward(loss1)
        opt1.step()
        
        r_x = self.forward2(d_x.detach())
        r_x = r_x.squeeze(1)
        iou2 = iou(r_x.round().bool(), voxel_data.round().bool())
        loss2 = bce_loss(r_x, voxel_data) * 10
        opt2.zero_grad()
        self.manual_backward(loss2)
        opt2.step()
        
        self.log_dict({"d_loss": loss1, "r_loss": loss2}, prog_bar=True)
        self.log_dict({"d_iou": iou1, "r_iou": iou2}, prog_bar=True)
        self.log("train_loss",loss2)
        self.log("train_iou",iou2)
        
        return {"d_loss": loss1, "r_loss": loss2, "pred": r_x}
    
    def validation_step(self, batch, batch_idx):
        csi_data_img, voxel_data = batch
        d_x = self.forward1(csi_data_img)
        r_x = self.forward2(d_x)
        r_x = r_x.squeeze(1)
        iou_value = iou(r_x.round().bool(), voxel_data.round().bool())
        bce_loss = torch.nn.BCELoss()
        loss = bce_loss(r_x, voxel_data) * 10
        self.log('val_loss', loss)
        self.log('val_iou', iou_value, prog_bar=True)
        return {"loss" : loss, "pred": r_x}

    
    def test_step(self, batch, batch_idx):
        csi_data_img, voxel_data = batch
        d_x = self.forward1(csi_data_img)
        r_x = self.forward2(d_x)
        d_voxel = d_x.squeeze(1)
        r_voxel = r_x.squeeze(1)
        d_iou_value = iou(d_voxel.round().bool(), voxel_data.round().bool())
        r_iou_value = iou(r_voxel.round().bool(), voxel_data.round().bool())
        bce_loss = torch.nn.BCELoss()
        loss = bce_loss(r_voxel, voxel_data) * 10
        self.log('test_loss', loss, sync_dist=True)
        self.log('test_iou', r_iou_value, prog_bar=True, sync_dist=True)
        if not os.path.exists("vox_result"):
            os.makedirs("vox_result")
        idx_folder_path = os.path.join("vox_result", str(batch_idx))
        if not os.path.exists(idx_folder_path):
            os.makedirs(idx_folder_path)
        for i_ in range(d_voxel.shape[0]):
            html = html_grid([np.squeeze(d_voxel[i_,:,:,:].cpu().detach().numpy().round()),voxel_data[i_,:,:,:].cpu().numpy().round()], rows=1, cols=2, height=800)
            with open(f'{idx_folder_path}/d_val_output_{i_}.html', 'w') as file:
                file.write(html)
            html = html_grid([np.squeeze(r_voxel[i_,:,:,:].cpu().detach().numpy().round()),voxel_data[i_,:,:,:].cpu().numpy().round()], rows=1, cols=2, height=800)
            with open(f'{idx_folder_path}/r_val_output_{i_}.html', 'w') as file:
                file.write(html)
        return {"loss" : loss, "pred": d_voxel}

        
    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam([{"params":self.encoder.parameters()},{"params":self.decoder.parameters()}], lr=1e-3, betas=(.9, .999),weight_decay=0.000001)
        optimizer2 = torch.optim.Adam([{"params":self.refiner.parameters()}], lr=1e-3, betas=(.9, .999),weight_decay=0.000001) 
        lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1,milestones=[150],gamma=.5)
        lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2,milestones=[150],gamma=.5)
        return [optimizer1,optimizer2],[lr_scheduler1,lr_scheduler2]