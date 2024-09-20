from models.model import Model
from tools.loss import *
from tools.uncertainty import *
from tools.geometry import *
import einops

class EvidentialTopK(Model):
    def __init__(self, *args, **kwargs):
        super(EvidentialTopK, self).__init__(*args, **kwargs)

        # !! Will be overwrite after the model is created in train.py !!
        self.beta_lambda = 0.001
        self.ood_lambda = 0.01
        self.k = 64

    @staticmethod
    def aleatoric(alpha, mode='dissonance'):
        if mode == 'aleatoric':
            soft = EvidentialTopK.activate(alpha)
            max_soft, hard = soft.max(dim=1)
            return (1 - max_soft).unsqueeze(1)
        elif mode == 'dissonance':
            return dissonance(alpha)

    @staticmethod
    def epistemic(alpha):
        return vacuity(alpha)

    @staticmethod
    def activate(alpha):
        return alpha / torch.sum(alpha, dim=1, keepdim=True)

    def loss(self, alpha, y, reduction='mean'):
        if self.loss_type == 'ce':
            A = uce_loss(alpha, y, weights=self.weights) + entropy_reg(alpha, beta_reg=self.beta_lambda)
        elif self.loss_type == 'focal':
            A = u_focal_loss(alpha, y, weights=self.weights, n=self.gamma)
        else:
            raise NotImplementedError()

        if reduction == 'mean':
            return A.mean()
        else:
            return A

    def loss_ood(self, alpha, y, ood, mapped_uncertainty, K=40):
        A = self.loss(alpha, y, reduction='none')
        # A *= 1 + (self.epistemic(alpha).detach() * self.k)
        A = A[~ood.bool()].mean()

        epistemic = self.epistemic(alpha)
        # print(alpha.shape, epistemic.shape, mapped_uncertainty.shape)
        top_k, top_k_idx = torch.topk(epistemic.reshape(epistemic.shape[0], -1), K)
        # print(top_k_idx.shape)
        # oreg = ood_reg(alpha, ood) * self.ood_lambda
        # print('alpha', alpha.reshape(alpha.shape[0], alpha.shape[1], -1).shape)
        # print('mapped_uncertainty', mapped_uncertainty.reshape(mapped_uncertainty.shape[0], -1).shape)

        # Workaround torch gather bugs that's not fixed until 2.0.0: https://github.com/pytorch/pytorch/issues/99595
        # reg_idx = einops.repeat(top_k_idx, 'b k -> b c k', c=alpha.shape[1])
        # flat_alpha = alpha.reshape(alpha.shape[0], alpha.shape[1], -1)
        # reg_alpha = []
        # for i in range(flat_alpha.shape[0]):
        #     reg_alpha.append([])
        #     for j in range(flat_alpha.shape[1]):
        #         reg_alpha[i].append(flat_alpha[i][j][reg_idx[i][j]])
        #     reg_alpha[i] = torch.stack(reg_alpha[i])
        # reg_alpha = torch.stack(reg_alpha)
        
        reg_alpha = torch.gather(alpha.reshape(alpha.shape[0], alpha.shape[1], -1), 2, einops.repeat(top_k_idx, 'b k -> b c k', c=alpha.shape[1]))
        reg_mapped_uncertainty = torch.gather(mapped_uncertainty.reshape(mapped_uncertainty.shape[0], -1), 1, top_k_idx)
        # print(reg_alpha.shape)
        # print(reg_mapped_uncertainty.shape)
        oreg = ood_reg_topk(reg_alpha, reg_mapped_uncertainty) * self.ood_lambda
        A += oreg

        return A, oreg

    def train_step_ood(self, images, intrinsics, extrinsics, labels, ood, mapped_uncertainty):
        self.opt.zero_grad(set_to_none=True)

        outs = self(images, intrinsics, extrinsics)
        loss, oodl = self.loss_ood(outs, labels.to(self.device), ood, mapped_uncertainty.to(self.device))
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.opt.step()

        preds = self.activate(outs)
        return outs, preds, loss, oodl

    def forward(self, images, intrinsics, extrinsics, limit=None):
        if self.tsne:
            print("Returning intermediate")
            return self.backbone(images, intrinsics, extrinsics)

        evidence = self.backbone(images, intrinsics, extrinsics).relu()

        if limit is not None:
            evidence = evidence.clamp(max=limit)
        alpha = evidence + 1

        return alpha

