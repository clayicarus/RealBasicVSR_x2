import torch
import torch.nn as nn
from torch.nn import functional as F

from mmedit.models.backbones.sr_backbones.basicvsr_net import (SPyNet, ResidualBlocksWithInputConv)
from mmedit.models.restorers import RealESRGAN
from mmedit.models.registry import BACKBONES, MODELS
from mmedit.models.builder import build_loss
from mmedit.models.common import set_requires_grad
from mmedit.utils import get_root_logger

from mmcv.runner import load_checkpoint
from mmcv.parallel import is_module_wrapper

from mmedit.models.common import (PixelShufflePack, flow_warp)

@BACKBONES.register_module()
class BasicVSRNet_x2(nn.Module):
    """BasicVSR network structure for video super-resolution.

    Support only x2 upsampling.
    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 64.
        num_blocks (int): Number of residual blocks in each propagation branch.
            Default: 30.
        spynet_pretrained (str): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self, mid_channels=64, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.mid_channels = mid_channels

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        # propagation branches
        self.backward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)

        # upsample
        self.fusion = nn.Conv2d(
            mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        # self.upsample1 = PixelShufflePack(
        #     mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)  # out_scale == 2 * 1
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)  # out_scale == 2

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, (
            'The height and width of inputs should be at least 64, '
            f'but got {h} and {w}.')

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)

        # compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)

        # backward-time propagation
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)

            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            # upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            # out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lr_curr)
            out += base
            outputs[i] = out

        res = torch.stack(outputs, dim=1)
        return res

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


@BACKBONES.register_module()
class RealBasicVSRNet_x2(nn.Module):
    """RealBasicVSR network structure for real-world video super-resolution.

    Support only x2 upsampling.
    Paper:
        Investigating Tradeoffs in Real-World Video Super-Resolution, arXiv

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_propagation_blocks (int, optional): Number of residual blocks in
            each propagation branch. Default: 20.
        num_cleaning_blocks (int, optional): Number of residual blocks in the
            image cleaning module. Default: 20.
        dynamic_refine_thres (int, optional): Stop cleaning the images when
            the residue is smaller than this value. Default: 255.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
        is_fix_cleaning (bool, optional): Whether to fix the weights of
            the image cleaning module during training. Default: False.
        is_sequential_cleaning (bool, optional): Whether to clean the images
            sequentially. This is used to save GPU memory, but the speed is
            slightly slower. Default: False.
    """

    def __init__(self,
                 mid_channels=64,
                 num_propagation_blocks=20,
                 num_cleaning_blocks=20,
                 dynamic_refine_thres=255,
                 spynet_pretrained=None,
                 is_fix_cleaning=False,
                 is_sequential_cleaning=False):

        super().__init__()

        self.dynamic_refine_thres = dynamic_refine_thres / 255.
        self.is_sequential_cleaning = is_sequential_cleaning

        # image cleaning module
        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, mid_channels, num_cleaning_blocks),
            nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True),
        )

        if is_fix_cleaning:  # keep the weights of the cleaning module fixed
            self.image_cleaning.requires_grad_(False)

        # BasicVSR
        self.basicvsr = BasicVSRNet_x2(mid_channels, num_propagation_blocks,
                                    spynet_pretrained)
        self.basicvsr.spynet.requires_grad_(False)

    def forward(self, lqs, return_lqs=False):
        n, t, c, h, w = lqs.size()
        for _ in range(0, 3):  # at most 3 cleaning, determined empirically
            if self.is_sequential_cleaning:
                residues = []
                for i in range(0, t):
                    residue_i = self.image_cleaning(lqs[:, i, :, :, :])
                    lqs[:, i, :, :, :] += residue_i
                    residues.append(residue_i)
                residues = torch.stack(residues, dim=1)
            else:  # time -> batch, then apply cleaning at once
                lqs = lqs.view(-1, c, h, w)
                residues = self.image_cleaning(lqs)
                lqs = (lqs + residues).view(n, t, c, h, w)

            # determine whether to continue cleaning
            if torch.mean(torch.abs(residues)) < self.dynamic_refine_thres:
                break

        # Super-resolution (BasicVSR)
        outputs = self.basicvsr(lqs)

        if return_lqs:
            return outputs, lqs
        else:
            return outputs

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


@MODELS.register_module()
class RealBasicVSR_x2(RealESRGAN):
    """RealBasicVSR model for real-world video super-resolution.

    Ref:
    Investigating Tradeoffs in Real-World Video Super-Resolution, arXiv

    Args:
        generator (dict): Config for the generator.
        discriminator (dict, optional): Config for the discriminator.
            Default: None.
        gan_loss (dict, optional): Config for the gan loss.
            Note that the loss weight in gan loss is only for the generator.
        pixel_loss (dict, optional): Config for the pixel loss. Default: None.
        cleaning_loss (dict, optional): Config for the image cleaning loss.
            Default: None.
        perceptual_loss (dict, optional): Config for the perceptual loss.
            Default: None.
        is_use_sharpened_gt_in_pixel (bool, optional): Whether to use the image
            sharpened by unsharp masking as the GT for pixel loss.
            Default: False.
        is_use_sharpened_gt_in_percep (bool, optional): Whether to use the
            image sharpened by unsharp masking as the GT for perceptual loss.
            Default: False.
        is_use_sharpened_gt_in_gan (bool, optional): Whether to use the
            image sharpened by unsharp masking as the GT for adversarial loss.
            Default: False.
        is_use_ema (bool, optional): When to apply exponential moving average
            on the network weights. Default: True.
        train_cfg (dict): Config for training. Default: None.
            You may change the training of gan by setting:
            `disc_steps`: how many discriminator updates after one generate
            update;
            `disc_init_steps`: how many discriminator updates at the start of
            the training.
            These two keys are useful when training with WGAN.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 discriminator=None,
                 gan_loss=None,
                 pixel_loss=None,
                 cleaning_loss=None,
                 perceptual_loss=None,
                 is_use_sharpened_gt_in_pixel=False,
                 is_use_sharpened_gt_in_percep=False,
                 is_use_sharpened_gt_in_gan=False,
                 is_use_ema=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super().__init__(generator, discriminator, gan_loss, pixel_loss,
                         perceptual_loss, is_use_sharpened_gt_in_pixel,
                         is_use_sharpened_gt_in_percep,
                         is_use_sharpened_gt_in_gan, is_use_ema, train_cfg,
                         test_cfg, pretrained)

        self.cleaning_loss = build_loss(
            cleaning_loss) if cleaning_loss else None

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """

        # during initialization, load weights from the ema model
        if (self.step_counter == self.start_iter
                and self.generator_ema is not None):
            if is_module_wrapper(self.generator):
                self.generator.module.load_state_dict(
                    self.generator_ema.module.state_dict())
            else:
                self.generator.load_state_dict(self.generator_ema.state_dict())

        # data
        lq = data_batch['lq']  # xi
        gt = data_batch['gt']  # ground-truth

        gt_pixel, gt_percep, gt_gan = gt.clone(), gt.clone(), gt.clone()
        if self.is_use_sharpened_gt_in_pixel:
            gt_pixel = data_batch['gt_unsharp']
        if self.is_use_sharpened_gt_in_percep:
            gt_percep = data_batch['gt_unsharp']
        if self.is_use_sharpened_gt_in_gan:
            gt_gan = data_batch['gt_unsharp']

        if self.cleaning_loss:
            n, t, c, h, w = gt.size()
            gt_clean = gt_pixel.view(-1, c, h, w)
            gt_clean = F.interpolate(gt_clean, scale_factor=0.5, mode='area')  # scale 0.5
            gt_clean = gt_clean.view(n, t, c, h // 2, w // 2)  # scale 1/2

        # generator
        fake_g_output, fake_g_lq = self.generator(lq, return_lqs=True)  # yi = S(xi), yi: (n, t, c, 2h, 2w)
        losses = dict()
        log_vars = dict()

        # reshape: (n, t, c, h, w) -> (n*t, c, h, w)
        c, h, w = gt.shape[2:]
        gt_pixel = gt_pixel.view(-1, c, h, w)
        gt_percep = gt_percep.view(-1, c, h, w)
        gt_gan = gt_gan.view(-1, c, h, w)
        fake_g_output = fake_g_output.view(-1, c, h, w)

        # no updates to discriminator parameters
        if self.gan_loss:
            set_requires_grad(self.discriminator, False)

        if (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps):
            if self.pixel_loss:
                losses['loss_pix'] = self.pixel_loss(fake_g_output, gt_pixel)
            if self.cleaning_loss:
                losses['loss_clean'] = self.cleaning_loss(fake_g_lq, gt_clean)
            if self.perceptual_loss:
                loss_percep, loss_style = self.perceptual_loss(
                    fake_g_output, gt_percep)
                if loss_percep is not None:
                    losses['loss_perceptual'] = loss_percep
                if loss_style is not None:
                    losses['loss_style'] = loss_style

            # gan loss for generator
            if self.gan_loss:
                fake_g_pred = self.discriminator(fake_g_output)
                losses['loss_gan'] = self.gan_loss(
                    fake_g_pred, target_is_real=True, is_disc=False)

            # parse loss
            loss_g, log_vars_g = self.parse_losses(losses)
            log_vars.update(log_vars_g)

            # optimize
            optimizer['generator'].zero_grad()
            loss_g.backward()
            optimizer['generator'].step()

        # discriminator
        if self.gan_loss:
            set_requires_grad(self.discriminator, True)
            # real
            real_d_pred = self.discriminator(gt_gan)
            loss_d_real = self.gan_loss(
                real_d_pred, target_is_real=True, is_disc=True)
            loss_d, log_vars_d = self.parse_losses(
                dict(loss_d_real=loss_d_real))
            optimizer['discriminator'].zero_grad()
            loss_d.backward()
            log_vars.update(log_vars_d)

            # fake
            fake_d_pred = self.discriminator(fake_g_output.detach())
            loss_d_fake = self.gan_loss(
                fake_d_pred, target_is_real=False, is_disc=True)
            loss_d, log_vars_d = self.parse_losses(
                dict(loss_d_fake=loss_d_fake))
            loss_d.backward()
            log_vars.update(log_vars_d)

            optimizer['discriminator'].step()

        self.step_counter += 1

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=fake_g_output.cpu()))

        return outputs

