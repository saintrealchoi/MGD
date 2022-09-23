import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES

@DISTILL_LOSSES.register_module()
class FeatureLoss(nn.Module):

    """PyTorch version of `Masked Generative Distillation`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.65
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 alpha_mgd=0.00002,
                 beta_mgd=0.00001,
                 lambda_mgd=0.65,
                 ):
        super(FeatureLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.beta_mgd = beta_mgd
        self.lambda_mgd = lambda_mgd
        self.name = name
    
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))

        self.feature_alignment = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))

        self.bg_generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))


    def forward(self,
                preds_S,
                preds_T,
                gt_bboxes,
                img_metas,
                loss_name):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)

        if loss_name.split('_')[2] == 'align':
            loss = self.get_align_loss(preds_S, preds_T)*self.alpha_mgd
        elif loss_name.split('_')[2] == 'bg':
            loss = self.get_bg_dis_loss(preds_S, preds_T, gt_bboxes, img_metas, loss_name)*self.beta_mgd
        else:
            loss = self.get_dis_loss(preds_S, preds_T, gt_bboxes, img_metas, loss_name)*self.alpha_mgd

        return loss

    def get_dis_loss(self, preds_S, preds_T, gt_bboxes, img_metas, loss_name):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat>1-self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T)/N

        return dis_loss

    def get_align_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        new_fea = self.feature_alignment(preds_S)

        align_loss = loss_mse(new_fea, preds_T)/N

        return align_loss
    
    def get_bg_dis_loss(self, preds_S, preds_T, gt_bboxes, img_metas, loss_name):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device

        Mask = torch.ones(N,H,W).to(device)
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i]).to(device)
            new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

            new_boxxes = new_boxxes.int()

            for k in range(len(new_boxxes)):
                Mask[i][new_boxxes[k][1]:new_boxxes[k][3]][new_boxxes[k][0]:new_boxxes[k][2]]=0

        Mask = Mask.unsqueeze(dim=1)

        masked_fea = torch.mul(preds_S, Mask)
        new_fea = self.bg_generation(masked_fea)
        new_fea = torch.mul(new_fea, Mask)
        preds_T = torch.mul(preds_T, Mask)

        bg_dis_loss = loss_mse(new_fea, preds_T)/N

        return bg_dis_loss