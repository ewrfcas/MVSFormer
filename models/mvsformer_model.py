import math
import os

import models.gvt as gvts
import models.vision_transformer as vits
from models.module import *
from models.warping import homo_warping_3D_with_mask

Align_Corners_Range = False


class identity_with(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


autocast = torch.cuda.amp.autocast if torch.__version__ >= '1.6.0' else identity_with


class StageNet(nn.Module):
    def __init__(self, args, ndepth, stage_idx):
        super(StageNet, self).__init__()
        self.args = args
        self.fusion_type = args.get('fusion_type', 'cnn')
        self.ndepth = ndepth
        self.stage_idx = stage_idx

        in_channels = args['base_ch']
        if self.fusion_type == 'cnn':
            model_th = args.get('model_th', 8)
            self.vis = nn.Sequential(ConvBnReLU(1, 16), ConvBnReLU(16, 16), ConvBnReLU(16, 8), nn.Conv2d(8, 1, 1), nn.Sigmoid())
            if ndepth <= model_th:
                self.cost_reg = CostRegNet3D(in_channels, args['base_ch'])
            else:
                self.cost_reg = CostRegNet(in_channels, args['base_ch'])
        elif self.fusion_type == 'epipole':
            self.attn_temp = args.get('attn_temp', 2.0)
            self.cost_reg = CostRegNet2D(in_channels, args['base_ch'])
        elif self.fusion_type == 'epipoleV2':
            self.attn_temp = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
            self.cost_reg = CostRegNet3D(in_channels, args['base_ch'])
        else:
            raise NotImplementedError

    def forward(self, features, proj_matrices, depth_values, tmp=2.0):
        ref_feat = features[:, 0]
        src_feats = features[:, 1:]
        src_feats = torch.unbind(src_feats, dim=1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(src_feats) == len(proj_matrices) - 1, "Different number of images and projection matrices"

        # step 1. feature extraction
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        volume_sum = 0.0
        vis_sum = 0.0
        similarities = []
        with autocast(enabled=False):
            for src_feat, src_proj in zip(src_feats, src_projs):
                # warpped features
                src_feat = src_feat.to(torch.float32)
                src_proj_new = src_proj[:, 0].clone()
                src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
                ref_proj_new = ref_proj[:, 0].clone()
                ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
                warped_volume, proj_mask = homo_warping_3D_with_mask(src_feat, src_proj_new, ref_proj_new, depth_values)

                B, C, D, H, W = warped_volume.shape
                G = self.args['base_ch']
                warped_volume = warped_volume.view(B, G, C // G, D, H, W)
                ref_volume = ref_feat.view(B, G, C // G, 1, H, W).repeat(1, 1, 1, D, 1, 1).to(torch.float32)
                in_prod_vol = (ref_volume * warped_volume).mean(dim=2)  # [B,G,D,H,W]

                if not self.training:
                    similarity = F.normalize(ref_volume, dim=1) * F.normalize(warped_volume, dim=1)
                    similarity = similarity.mean(dim=2)
                    similarity = similarity.sum(dim=1)
                    similarities.append(similarity.unsqueeze(1))

                if self.fusion_type == 'cnn':
                    sim_vol = in_prod_vol.sum(dim=1)  # [B,D,H,W]
                    sim_vol_norm = F.softmax(sim_vol.detach(), dim=1)
                    entropy = (- sim_vol_norm * torch.log(sim_vol_norm + 1e-7)).sum(dim=1, keepdim=True)
                    vis_weight = self.vis(entropy)
                elif self.fusion_type == 'epipole':
                    vis_weight = torch.softmax(in_prod_vol.sum(1) / self.attn_temp, dim=1) / math.sqrt(C)  # B D H W
                elif self.fusion_type == 'epipoleV2':
                    attn_score = in_prod_vol.sum(1) / torch.clamp(self.attn_temp, 0.1, 10.)  # [B,D,H,W]
                    attn_score = attn_score + (-10000.0 * proj_mask)
                    vis_weight = torch.softmax(attn_score, dim=1) / math.sqrt(G)  # B D H W
                else:
                    raise NotImplementedError

                volume_sum = volume_sum + in_prod_vol * vis_weight.unsqueeze(1)
                vis_sum = vis_sum + vis_weight

            # aggregate multiple feature volumes by variance
            volume_mean = volume_sum / (vis_sum.unsqueeze(1) + 1e-6)  # volume_sum / (num_views - 1)

        # step 3. cost volume regularization
        cost_reg = self.cost_reg(volume_mean)

        prob_volume_pre = cost_reg.squeeze(1)
        prob_volume = F.softmax(prob_volume_pre, dim=1)

        if self.args['depth_type'] == 'ce' or self.args['depth_type'] == 'was':
            if type(tmp) == list:
                tmp = tmp[self.stage_idx]

            if self.training:
                _, idx = torch.max(prob_volume, dim=1)
                # vanilla argmax
                depth = torch.gather(depth_values, dim=1, index=idx.unsqueeze(1)).squeeze(1)
            else:
                # regression (t)
                depth = depth_regression(F.softmax(prob_volume_pre * tmp, dim=1), depth_values=depth_values)
            # conf
            photometric_confidence = prob_volume.max(1)[0]  # [B,H,W]
        elif self.args['depth_type'] == 'mixup_ce':
            prob_left = prob_volume[:, :-1]  # [B,D-1,H,W]
            prob_right = prob_volume[:, 1:]  # [B,D-1,H,W]
            mixup_prob = prob_left + prob_right  # [B,D-1,H,W]
            photometric_confidence, idx = torch.max(mixup_prob, dim=1)  # [B,H,W]
            # 我们假设inverse depth range中间是线性的, 重归一化
            prob_left_right_sum = prob_left + prob_right + 1e-7
            prob_left_normed = prob_left / prob_left_right_sum
            prob_right_normed = prob_right / prob_left_right_sum
            mixup_depth = depth_values[:, :-1] * prob_left_normed + depth_values[:, 1:] * prob_right_normed  # [B,D-1,H,W]
            depth = torch.gather(mixup_depth, dim=1, index=idx.unsqueeze(1)).squeeze(1)
        else:
            depth = depth_regression(prob_volume, depth_values=depth_values)
            if self.ndepth >= 32:
                photometric_confidence = conf_regression(prob_volume, n=4)
            elif self.ndepth == 16:
                photometric_confidence = conf_regression(prob_volume, n=3)
            elif self.ndepth == 8:
                photometric_confidence = conf_regression(prob_volume, n=2)
            else:
                photometric_confidence = prob_volume.max(1)[0]  # [B,H,W]

        outputs = {'depth': depth, 'prob_volume': prob_volume, "photometric_confidence": photometric_confidence.detach(),
                   'depth_values': depth_values, 'prob_volume_pre': prob_volume_pre}

        if not self.training:
            try:
                similarities = torch.sum(torch.cat(similarities, dim=1), dim=1)
                sim_idx = torch.argmax(similarities, dim=1).unsqueeze(1)
                sim_depth = torch.gather(depth_values, index=sim_idx, dim=1).squeeze(1)
                outputs['sim_depth'] = sim_depth
            except:
                outputs['sim_depth'] = torch.zeros_like(depth)

        return outputs


class DINOMVSNet(nn.Module):
    def __init__(self, args):
        super(DINOMVSNet, self).__init__()
        self.args = args
        self.ndepths = args['ndepths']
        self.depth_interals_ratio = args['depth_interals_ratio']
        self.inverse_depth = args.get('inverse_depth', False)
        self.multi_scale = args.get('multi_scale', False)

        self.encoder = FPNEncoder(feat_chs=args['feat_chs'])
        if self.multi_scale:
            self.decoder = FPNDecoderV2(feat_chs=args['feat_chs'])
        else:
            self.decoder = FPNDecoder(feat_chs=args['feat_chs'])

        self.do_vit = True
        self.vit_args = args['vit_args']
        self.vit = vits.__dict__[self.vit_args['vit_arch']](patch_size=self.vit_args['patch_size'],
                                                            qk_scale=self.vit_args['qk_scale'])
        if os.path.exists(self.vit_args['vit_path']):
            state_dict = torch.load(self.vit_args['vit_path'], map_location='cpu')
            from utils import torch_init_model
            if self.vit_args['vit_path'].split('/')[-1] == 'model_best.pth' and 'state_dict' in state_dict:
                state_dict_ = state_dict['state_dict']
                state_dict = {}
                for k in state_dict_:
                    if k.startswith('vit.'):
                        state_dict[k.replace('vit.', '')] = state_dict_[k]
            torch_init_model(self.vit, state_dict, key='model')
        else:
            print('!!!No weight in', self.vit_args['vit_path'], 'testing should neglect this.')

        if not self.vit_args['att_fusion']:
            self.decoder_vit = VITDecoderStage4NoAtt(self.vit_args)
        else:
            if self.multi_scale:
                self.decoder_vit = VITDecoderStage4(self.vit_args)
            else:
                self.decoder_vit = VITDecoderStage4Single(self.vit_args)

        self.fusions = nn.ModuleList([StageNet(args, self.ndepths[i], i) for i in range(len(self.ndepths))])

    def forward(self, imgs, proj_matrices, depth_values, tmp=2.0):
        B, V, H, W = imgs.shape[0], imgs.shape[1], imgs.shape[3], imgs.shape[4]
        depth_interval = depth_values[:, 1] - depth_values[:, 0]

        if self.training:
            # feature encode
            imgs = imgs.reshape(B * V, 3, H, W)
            conv01, conv11, conv21, conv31 = self.encoder(imgs)

            vit_h, vit_w = int(H * self.vit_args['rescale']), int(W * self.vit_args['rescale'])
            vit_imgs = F.interpolate(imgs, (vit_h, vit_w), mode='bicubic', align_corners=Align_Corners_Range)
            if self.args['fix']:
                with torch.no_grad():
                    vit_feat, vit_att = self.vit.forward_with_last_att(vit_imgs)
            else:
                vit_feat, vit_att = self.vit.forward_with_last_att(vit_imgs)
            vit_feat = vit_feat[:, 1:].reshape(B * V, vit_h // self.vit_args['patch_size'], vit_w // self.vit_args['patch_size'],
                                               self.vit_args['vit_ch']).permute(0, 3, 1, 2).contiguous()  # [BV,C,h,w]
            vit_att = vit_att[:, :, 0, 1:].reshape(B * V, -1, vit_h // self.vit_args['patch_size'], vit_w // self.vit_args['patch_size'])
            if self.multi_scale:
                vit_out, vit_out2, vit_out3 = self.decoder_vit.forward(vit_feat, vit_att)
                feat1, feat2, feat3, feat4 = self.decoder.forward(conv01, conv11, conv21, conv31, vit_out, vit_out2, vit_out3)
            else:
                vit_out = self.decoder_vit.forward(vit_feat, vit_att)
                conv31 = conv31 + vit_out
                feat1, feat2, feat3, feat4 = self.decoder.forward(conv01, conv11, conv21, conv31)

            features = {'stage1': feat1.reshape(B, V, feat1.shape[1], feat1.shape[2], feat1.shape[3]),
                        'stage2': feat2.reshape(B, V, feat2.shape[1], feat2.shape[2], feat2.shape[3]),
                        'stage3': feat3.reshape(B, V, feat3.shape[1], feat3.shape[2], feat3.shape[3]),
                        'stage4': feat4.reshape(B, V, feat4.shape[1], feat4.shape[2], feat4.shape[3])}
        else:
            feat1s, feat2s, feat3s, feat4s = [], [], [], []
            for vi in range(V):
                img_v = imgs[:, vi]
                conv01, conv11, conv21, conv31 = self.encoder(img_v)

                vit_h, vit_w = int(H * self.vit_args['rescale']), int(W * self.vit_args['rescale'])
                vit_imgs = F.interpolate(img_v, (vit_h, vit_w), mode='bicubic', align_corners=Align_Corners_Range)
                if self.args['fix']:
                    with torch.no_grad():
                        vit_feat, vit_att = self.vit.forward_with_last_att(vit_imgs)
                else:
                    vit_feat, vit_att = self.vit.forward_with_last_att(vit_imgs)

                vit_feat = vit_feat[:, 1:].reshape(B, vit_h // self.vit_args['patch_size'], vit_w // self.vit_args['patch_size'],
                                                   self.vit_args['vit_ch']).permute(0, 3, 1, 2).contiguous()  # [B,C,h,w]
                # [B,nh,hw-1]
                vit_att = vit_att[:, :, 0, 1:].reshape(B, -1, vit_h // self.vit_args['patch_size'], vit_w // self.vit_args['patch_size'])

                if self.multi_scale:
                    vit_out, vit_out2, vit_out3 = self.decoder_vit.forward(vit_feat, vit_att)
                    feat1, feat2, feat3, feat4 = self.decoder.forward(conv01, conv11, conv21, conv31, vit_out, vit_out2, vit_out3)
                else:
                    vit_out = self.decoder_vit.forward(vit_feat, vit_att)
                    conv31 = conv31 + vit_out
                    feat1, feat2, feat3, feat4 = self.decoder.forward(conv01, conv11, conv21, conv31)

                feat1s.append(feat1)
                feat2s.append(feat2)
                feat3s.append(feat3)
                feat4s.append(feat4)

            features = {'stage1': torch.stack(feat1s, dim=1),
                        'stage2': torch.stack(feat2s, dim=1),
                        'stage3': torch.stack(feat3s, dim=1),
                        'stage4': torch.stack(feat4s, dim=1)}

        outputs = {}
        outputs_stage = {}

        prob_maps = torch.zeros([B, H, W], dtype=torch.float32, device=imgs.device)
        for stage_idx in range(len(self.ndepths)):
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            features_stage = features['stage{}'.format(stage_idx + 1)]
            B, V, C, H, W = features_stage.shape

            # init range
            if stage_idx == 0:
                if self.inverse_depth:
                    depth_samples = init_inverse_range(depth_values, self.ndepths[stage_idx], imgs.device, imgs.dtype, H, W)
                else:
                    depth_samples = init_range(depth_values, self.ndepths[stage_idx], imgs.device, imgs.dtype, H, W)
            else:
                if self.inverse_depth:
                    depth_samples = schedule_inverse_range(outputs_stage['depth'].detach(), outputs_stage['depth_values'],
                                                           self.ndepths[stage_idx], self.depth_interals_ratio[stage_idx], H, W)  # B D H W
                else:
                    depth_samples = schedule_range(outputs_stage['depth'].detach(), self.ndepths[stage_idx],
                                                   self.depth_interals_ratio[stage_idx] * depth_interval, H, W)

            outputs_stage = self.fusions[stage_idx].forward(features_stage, proj_matrices_stage, depth_samples, tmp=tmp)
            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            if outputs_stage['photometric_confidence'].shape[1] != prob_maps.shape[1] or outputs_stage['photometric_confidence'].shape[2] != prob_maps.shape[2]:
                outputs_stage['photometric_confidence'] = F.interpolate(outputs_stage['photometric_confidence'].unsqueeze(1),
                                                                        [prob_maps.shape[1], prob_maps.shape[2]], mode="nearest").squeeze(1)
            prob_maps += outputs_stage['photometric_confidence']
            outputs.update(outputs_stage)

        outputs['refined_depth'] = outputs_stage['depth']
        outputs['photometric_confidence'] = prob_maps / len(self.ndepths)
        del prob_maps

        return outputs


class TwinMVSNet(nn.Module):
    def __init__(self, args):
        super(TwinMVSNet, self).__init__()
        self.args = args
        self.ndepths = args['ndepths']
        self.depth_interals_ratio = args['depth_interals_ratio']
        self.inverse_depth = args.get('inverse_depth', False)
        self.multi_scale = args.get('multi_scale', False)

        self.encoder = FPNEncoder(feat_chs=args['feat_chs'])
        if self.multi_scale:
            self.decoder = FPNDecoderV2(feat_chs=args['feat_chs'])
        else:
            self.decoder = FPNDecoder(feat_chs=args['feat_chs'])

        self.do_vit = True
        self.vit_args = args['vit_args']
        if self.vit_args['vit_arch'] == 'alt_gvt_small':
            self.vit = gvts.alt_gvt_small()
        elif self.vit_args['vit_arch'] == 'alt_gvt_base':
            self.vit = gvts.alt_gvt_base()
        elif self.vit_args['vit_arch'] == 'alt_gvt_large':
            self.vit = gvts.alt_gvt_large()

        if os.path.exists(self.vit_args['vit_path']):
            state_dict = torch.load(self.vit_args['vit_path'], map_location='cpu')
            from utils import torch_init_model
            torch_init_model(self.vit, state_dict, key='none')
        else:
            print('!!!No weight in', self.vit_args['vit_path'], 'testing should neglect this.')

        if self.multi_scale:
            self.decoder_vit = TwinDecoderStage4V2(self.vit_args)
        else:
            self.decoder_vit = TwinDecoderStage4(self.vit_args)

        self.fusions = nn.ModuleList([StageNet(args, self.ndepths[i], i) for i in range(len(self.ndepths))])

    def forward(self, imgs, proj_matrices, depth_values, tmp=2.0):
        B, V, H, W = imgs.shape[0], imgs.shape[1], imgs.shape[3], imgs.shape[4]

        depth_interval = depth_values[:, 1] - depth_values[:, 0]

        if self.training:
            # feature encode
            imgs = imgs.reshape(B * V, 3, H, W)
            conv01, conv11, conv21, conv31 = self.encoder(imgs)

            vit_h, vit_w = int(H * self.vit_args['rescale']), int(W * self.vit_args['rescale'])
            vit_imgs = F.interpolate(imgs, (vit_h, vit_w), mode='bicubic', align_corners=Align_Corners_Range)
            if self.args['fix']:
                with torch.no_grad():
                    [vit1, vit2, vit3, vit4] = self.vit.forward_features(vit_imgs)
            else:
                [vit1, vit2, vit3, vit4] = self.vit.forward_features(vit_imgs)

            if self.multi_scale:
                vit_out, vit_out2, vit_out3 = self.decoder_vit.forward(vit1, vit2, vit3, vit4)
                feat1, feat2, feat3, feat4 = self.decoder.forward(conv01, conv11, conv21, conv31, vit_out, vit_out2, vit_out3)
            else:
                vit_out = self.decoder_vit.forward(vit1, vit2, vit3, vit4)
                conv31 = conv31 + vit_out
                feat1, feat2, feat3, feat4 = self.decoder.forward(conv01, conv11, conv21, conv31)

            features = {'stage1': feat1.reshape(B, V, feat1.shape[1], feat1.shape[2], feat1.shape[3]),
                        'stage2': feat2.reshape(B, V, feat2.shape[1], feat2.shape[2], feat2.shape[3]),
                        'stage3': feat3.reshape(B, V, feat3.shape[1], feat3.shape[2], feat3.shape[3]),
                        'stage4': feat4.reshape(B, V, feat4.shape[1], feat4.shape[2], feat4.shape[3])}
        else:
            feat1s, feat2s, feat3s, feat4s = [], [], [], []
            for vi in range(V):
                img_v = imgs[:, vi]
                conv01, conv11, conv21, conv31 = self.encoder(img_v)

                vit_h, vit_w = int(H * self.vit_args['rescale']), int(W * self.vit_args['rescale'])
                vit_imgs = F.interpolate(img_v, (vit_h, vit_w), mode='bicubic', align_corners=Align_Corners_Range)
                if self.args['fix']:
                    with torch.no_grad():
                        [vit1, vit2, vit3, vit4] = self.vit.forward_features(vit_imgs)
                else:
                    [vit1, vit2, vit3, vit4] = self.vit.forward_features(vit_imgs)

                if self.multi_scale:
                    vit_out, vit_out2, vit_out3 = self.decoder_vit.forward(vit1, vit2, vit3, vit4)
                    feat1, feat2, feat3, feat4 = self.decoder.forward(conv01, conv11, conv21, conv31, vit_out, vit_out2, vit_out3)
                else:
                    vit_out = self.decoder_vit.forward(vit1, vit2, vit3, vit4)
                    conv31 = conv31 + vit_out
                    feat1, feat2, feat3, feat4 = self.decoder.forward(conv01, conv11, conv21, conv31)
                feat1s.append(feat1)
                feat2s.append(feat2)
                feat3s.append(feat3)
                feat4s.append(feat4)

            features = {'stage1': torch.stack(feat1s, dim=1),
                        'stage2': torch.stack(feat2s, dim=1),
                        'stage3': torch.stack(feat3s, dim=1),
                        'stage4': torch.stack(feat4s, dim=1)}

        outputs = {}
        outputs_stage = {}

        if self.args['depth_type'] in ['ce', 'mixup_ce']:
            prob_maps = torch.zeros([B, H, W], dtype=torch.float32, device=imgs.device)
        else:
            prob_maps = torch.empty(0)
        for stage_idx in range(len(self.ndepths)):
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            features_stage = features['stage{}'.format(stage_idx + 1)]
            B, V, C, H, W = features_stage.shape

            # init range
            if stage_idx == 0:
                if self.inverse_depth:
                    depth_samples = init_inverse_range(depth_values, self.ndepths[stage_idx], imgs.device, imgs.dtype, H, W)
                else:
                    depth_samples = init_range(depth_values, self.ndepths[stage_idx], imgs.device, imgs.dtype, H, W)
            else:
                if self.inverse_depth:
                    depth_samples = schedule_inverse_range(outputs_stage['depth'].detach(), outputs_stage['depth_values'],
                                                           self.ndepths[stage_idx], self.depth_interals_ratio[stage_idx], H, W)  # B D H W
                else:
                    depth_samples = schedule_range(outputs_stage['depth'].detach(), self.ndepths[stage_idx],
                                                   self.depth_interals_ratio[stage_idx] * depth_interval, H, W)

            outputs_stage = self.fusions[stage_idx].forward(features_stage, proj_matrices_stage, depth_samples, tmp=tmp)
            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            if self.args['depth_type'] in ['ce', 'mixup_ce']:
                if outputs_stage['photometric_confidence'].shape[1] != prob_maps.shape[1] or outputs_stage['photometric_confidence'].shape[2] != prob_maps.shape[2]:
                    outputs_stage['photometric_confidence'] = F.interpolate(outputs_stage['photometric_confidence'].unsqueeze(1),
                                                                            [prob_maps.shape[1], prob_maps.shape[2]], mode="nearest").squeeze(1)
                prob_maps += outputs_stage['photometric_confidence']
            outputs.update(outputs_stage)

        outputs['refined_depth'] = outputs_stage['depth']
        if self.args['depth_type'] in ['ce', 'mixup_ce']:
            outputs['photometric_confidence'] = prob_maps / len(self.ndepths)

        return outputs
