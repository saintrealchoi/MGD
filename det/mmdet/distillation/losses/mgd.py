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
                 lambda_mgd=0.65,
                 ):
        super(FeatureLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
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
            loss_name: Feature Loss name for FPN Layer
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)
    
        loss = self.get_dis_loss(preds_S, preds_T, gt_bboxes, img_metas, loss_name)*self.alpha_mgd
            
        return loss

    def get_dis_loss(self, preds_S, preds_T, gt_bboxes, img_metas, loss_name):
        loss_mse = nn.MSELoss(reduction='sum')
        N, _, H, W = preds_T.shape

        device = preds_S.device

        if loss_name[-1] == '3' or loss_name[-1] == '4':
            new_fea = self.generation(preds_S)
            dis_loss = loss_mse(new_fea, preds_T)/N
            return dis_loss
            
        else:
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
            mat_rand = torch.rand((N,1,H,W)).to(device)
            mat_rand = torch.where(mat_rand>1-self.lambda_mgd, 0, 1).to(device)
            mat_final = torch.logical_or(Mask,mat_rand).int()

            masked_fea = torch.mul(preds_S, mat_final)
            new_fea = self.generation(masked_fea)

            dis_loss = loss_mse(new_fea, preds_T)/N

        return dis_loss