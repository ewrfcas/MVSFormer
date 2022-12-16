import torch
import torch.nn.functional as F


class identity_with(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


autocast = torch.cuda.amp.autocast if torch.__version__ >= '1.6.0' else identity_with

def simple_loss(outputs, depth_gt_ms, mask_ms):
    depth_est = outputs["depth"]
    depth_gt = depth_gt_ms
    mask = mask_ms
    mask = mask > 0.5

    depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

    return depth_loss


def reg_loss(inputs, depth_gt_ms, mask_ms, dlossw, depth_interval):
    loss_dict = {}
    depth_interval = depth_interval.unsqueeze(-1).unsqueeze(-1)
    depth_loss_weights = dlossw

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in ['stage1', 'stage2', 'stage3']]:
        depth_est = stage_inputs["depth"] / depth_interval
        depth_gt = depth_gt_ms[stage_key] / depth_interval
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            loss_dict[stage_key] = depth_loss_weights[stage_idx] * depth_loss
        else:
            loss_dict[stage_key] = depth_loss

    return loss_dict


def reg_loss_stage4(inputs, depth_gt_ms, mask_ms, dlossw, depth_interval, mask_out_range=False, inverse_depth=True):
    loss_dict = {}
    depth_interval = depth_interval.unsqueeze(-1).unsqueeze(-1)
    depth_loss_weights = dlossw

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in ['stage1', 'stage2', 'stage3', 'stage4']]:
        depth_est = stage_inputs["depth"] / depth_interval
        depth_gt = depth_gt_ms[stage_key] / depth_interval
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        if mask_out_range:
            depth_values = stage_inputs["depth_values"]  # (b, d, h, w)
            if inverse_depth:
                depth_values = torch.flip(depth_values, dims=[1])
            intervals = torch.abs(depth_values[:, 1:] - depth_values[:, :-1]) / 2  # [b,d-1,h,w]
            intervals = torch.cat([intervals, intervals[:, -1:]], dim=1)  # [b,d,h,w]
            min_depth_values = depth_values[:, 0] - intervals[:, 0, ]
            max_depth_values = depth_values[:, -1] + intervals[:, -1]
            depth_gt_ = depth_gt_ms[stage_key]
            out_of_range_left = (depth_gt_ < min_depth_values).to(torch.float32)
            out_of_range_right = (depth_gt_ > max_depth_values).to(torch.float32)
            out_of_range_mask = torch.clamp(out_of_range_left + out_of_range_right, 0, 1)
            in_range_mask = (1 - out_of_range_mask).to(torch.bool)
            mask = mask & in_range_mask

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            loss_dict[stage_key] = depth_loss_weights[stage_idx] * depth_loss
        else:
            loss_dict[stage_key] = depth_loss

    return loss_dict


def sinkhorn(gt_depth, hypo_depth, attn_weight, mask, iters, eps=1, continuous=False):
    """
    gt_depth: B H W
    hypo_depth: B D H W
    attn_weight: B D H W
    mask: B H W
    """
    B, D, H, W = attn_weight.shape
    if not continuous:
        D_map = torch.stack([torch.arange(-i, D - i, 1, dtype=torch.float32, device=gt_depth.device) for i in range(D)], dim=1).abs()
        D_map = D_map[None, None, :, :].repeat(B, H * W, 1, 1)  # B HW D D
        gt_indices = torch.abs(hypo_depth - gt_depth[:, None, :, :]).min(1)[1].squeeze(1).reshape(B * H * W, 1)  # BHW, 1
        gt_dist = torch.zeros_like(hypo_depth).permute(0, 2, 3, 1).reshape(B * H * W, D)
        gt_dist.scatter_add_(1, gt_indices, torch.ones([gt_dist.shape[0], 1], dtype=gt_dist.dtype, device=gt_dist.device))
        gt_dist = gt_dist.reshape(B, H * W, D)  # B HW D
    else:
        gt_dist = torch.zeros((B, H * W, D + 1), dtype=torch.float32, device=gt_depth.device, requires_grad=False)  # B HW D+1
        gt_dist[:, :, -1] = 1
        D_map = torch.zeros((B, D, D + 1), dtype=torch.float32, device=gt_depth.device, requires_grad=False)  # B D D+1
        D_map[:, :D, :D] = torch.stack([torch.arange(-i, D - i, 1, dtype=torch.float32, device=gt_depth.device) for i in range(D)], dim=1).abs().unsqueeze(0)  # B D D+1
        D_map = D_map[:, None, None, :, :].repeat(1, H, W, 1, 1)  # B H W D D+1
        itv = 1 / hypo_depth[:, 2, :, :] - 1 / hypo_depth[:, 1, :, :]  # B H W
        gt_bin_distance_ = (1 / gt_depth - 1 / hypo_depth[:, 0, :, :]) / itv  # B H W
        # FIXME hard code 100
        gt_bin_distance_[~mask] = 10

        gt_bin_distance = torch.stack([(gt_bin_distance_ - i).abs() for i in range(D)], dim=1).permute(0, 2, 3, 1)  # B H W D
        D_map[:, :, :, :, -1] = gt_bin_distance
        D_map = D_map.reshape(B, H * W, D, 1 + D)  # B HW D D+1

    pred_dist = attn_weight.permute(0, 2, 3, 1).reshape(B, H * W, D)  # B HW D

    # map to log space for stability
    log_mu = (gt_dist + 1e-12).log()
    log_nu = (pred_dist + 1e-12).log()  # B HW D or D+1

    u, v = torch.zeros_like(log_nu), torch.zeros_like(log_mu)
    for _ in range(iters):
        # scale v first then u to ensure row sum is 1, col sum slightly larger than 1
        v = log_mu - torch.logsumexp(D_map / eps + u.unsqueeze(3), dim=2)  # log(sum(exp()))
        u = log_nu - torch.logsumexp(D_map / eps + v.unsqueeze(2), dim=3)

    # convert back from log space, recover probabilities by normalization 2W
    T_map = (D_map / eps + u.unsqueeze(3) + v.unsqueeze(2)).exp()  # B HW D D
    loss = (T_map * D_map).reshape(B * H * W, -1)[mask.reshape(-1)].sum(-1).mean()

    return T_map, loss


def wasserstein_loss(inputs, depth_gt_ms, mask_ms, dlossw, ot_iter=10, ot_eps=1, ot_continous=False, inverse=True):
    total_loss = {}
    stage_ot_loss = []
    # range_err_ratio = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        hypo_depth = stage_inputs['depth_values']
        attn_weight = stage_inputs['prob_volume']
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        depth_gt = depth_gt_ms[stage_key]

        # # mask range
        # if inverse:
        #     depth_itv = (1 / hypo_depth[:, 2, :, :] - 1 / hypo_depth[:, 1, :, :]).abs()  # B H W
        #     mask_out_of_range = ((1 / hypo_depth - 1 / depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0  # B H W
        # else:
        #     depth_itv = (hypo_depth[:, 2, :, :] - hypo_depth[:, 1, :, :]).abs()  # B H W
        #     mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0  # B H W
        # range_err_ratio.append(mask_out_of_range[mask].float().mean())

        this_stage_ot_loss = sinkhorn(depth_gt, hypo_depth, attn_weight, mask, iters=ot_iter, eps=ot_eps, continuous=ot_continous)[1]

        stage_ot_loss.append(this_stage_ot_loss)
        total_loss[stage_key] = dlossw[stage_idx] * this_stage_ot_loss

    return total_loss  # , range_err_ratio


def bimodel_loss(inputs, depth_gt_ms, mask_ms, dlossw, depth_interval):
    loss_dict = {}
    depth_interval = depth_interval.unsqueeze(-1).unsqueeze(-1)
    depth_loss_weights = dlossw

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in ['stage1', 'stage2', 'stage3']]:
        depth0 = stage_inputs['depth0'].to(torch.float32) / depth_interval
        depth1 = stage_inputs['depth1'].to(torch.float32) / depth_interval
        depth_gt = depth_gt_ms[stage_key] / depth_interval
        sigma0 = stage_inputs['sigma0'].to(torch.float32)
        sigma1 = stage_inputs['sigma1'].to(torch.float32)
        pi0 = stage_inputs['pi0'].to(torch.float32)
        pi1 = stage_inputs['pi1'].to(torch.float32)
        dist0 = pi0 * 0.5 * torch.exp(-(torch.abs(depth_gt - depth0) / sigma0)) / sigma0
        dist1 = pi1 * 0.5 * torch.exp(-(torch.abs(depth_gt - depth1) / sigma1)) / sigma1

        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = -torch.log(dist0[mask] + dist1[mask] + 1e-8)
        depth_loss = depth_loss.mean()

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            loss_dict[stage_key] = depth_loss_weights[stage_idx] * depth_loss
        else:
            loss_dict[stage_key] = depth_loss

    return loss_dict


import numpy as np


def DpethGradLoss(depth_grad_logits, depth_grad_gt, depth_grad_mask):
    B, H, W = depth_grad_logits.shape
    RB = B
    loss = 0.0
    for i in range(B):
        depth_grad_logits_ = depth_grad_logits[i]
        depth_grad_gt_ = depth_grad_gt[i]
        if torch.sum(depth_grad_gt_) == 0:
            RB = RB - 1
            continue
        depth_grad_mask_ = depth_grad_mask[i]
        pos_logits = depth_grad_logits_[depth_grad_gt_ == 1]
        depth_grad_mask_ = depth_grad_mask_ - depth_grad_gt_
        N = pos_logits.shape[0]
        neg_logits = depth_grad_logits_[depth_grad_mask_ == 1]
        shuffle_idx = np.arange(neg_logits.shape[0])
        np.random.shuffle(shuffle_idx)
        neg_logits = neg_logits[shuffle_idx[:N]]
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
        bloss = F.binary_cross_entropy_with_logits(logits, target=labels, reduction='mean')
        loss += bloss

    loss = loss / (RB + 1e-7)
    loss = loss * 5

    return loss


def cvx_reg_loss(inputs, depth_gt, mask, dlossw, depth_interval):
    loss_dict = {}
    depth_interval = depth_interval.unsqueeze(-1).unsqueeze(-1)
    depth_loss_weights = dlossw

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in ['stage1', 'stage2', 'stage3']]:
        depth_est = stage_inputs["depth"] / depth_interval
        depth_gt_stage = F.interpolate(depth_gt.unsqueeze(1), size=(depth_est.shape[1], depth_est.shape[2]), mode='nearest').squeeze(1)
        mask_stage = F.interpolate(mask.unsqueeze(1), size=(depth_est.shape[1], depth_est.shape[2]), mode='nearest').squeeze(1)
        depth_gt_stage = depth_gt_stage / depth_interval
        mask_stage = mask_stage > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask_stage], depth_gt_stage[mask_stage], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            loss_dict[stage_key] = depth_loss_weights[stage_idx] * depth_loss
        else:
            loss_dict[stage_key] = depth_loss

    return loss_dict


def ce_loss(inputs, depth_gt_ms, mask_ms, dlossw):
    depth_loss_weights = dlossw

    loss_dict = {}
    for sub_stage_key in inputs:
        if 'stage' not in sub_stage_key:
            continue
        stage_inputs = inputs[sub_stage_key]
        stage_key = sub_stage_key.split('_')[0]
        depth_gt = depth_gt_ms[stage_key]
        depth_values = stage_inputs["depth_values"]  # (b, d, h, w)
        interval = stage_inputs["interval"]  # float
        prob_volume_pre = stage_inputs["prob_volume_pre"].to(torch.float32)
        mask = mask_ms[stage_key]
        mask = (mask > 0.5).to(torch.float32)

        depth_gt = depth_gt.unsqueeze(1)
        depth_gt_volume = depth_gt.expand_as(depth_values)  # (b, d, h, w)
        # |-|-|-|-|
        #   x x x x
        depth_values_right = depth_values + interval / 2
        out_of_range_left = (depth_gt < depth_values[:, 0:1, :, :]).to(torch.float32)
        out_of_range_right = (depth_gt > depth_values[:, -1:, :, :]).to(torch.float32)
        out_of_range_mask = torch.clamp(out_of_range_left + out_of_range_right, 0, 1)
        in_range_mask = 1 - out_of_range_mask
        final_mask = in_range_mask.squeeze(1) * mask
        gt_index_volume = (depth_values_right <= depth_gt_volume).to(torch.float32).sum(dim=1, keepdims=True).to(torch.long)
        gt_index_volume = torch.clamp_max(gt_index_volume, max=depth_values.shape[1] - 1).squeeze(1)

        depth_loss = F.cross_entropy(prob_volume_pre, gt_index_volume, reduction='none')
        depth_loss = torch.sum(depth_loss * final_mask) / (torch.sum(final_mask) + 1e-6)

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            loss_dict[sub_stage_key] = depth_loss_weights[stage_idx] * depth_loss
        else:
            loss_dict[sub_stage_key] = depth_loss

    return loss_dict


def focal_loss(preds, labels, gamma=2.0):  # [B,D,H,W], [B,H,W]
    labels = labels.unsqueeze(1)
    preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
    preds_softmax = torch.exp(preds_logsoft)  # softmax

    preds_softmax = preds_softmax.gather(1, labels)
    preds_logsoft = preds_logsoft.gather(1, labels)
    loss = -torch.mul(torch.pow((1 - preds_softmax), gamma), preds_logsoft)

    return loss


def ce_loss_stage4(inputs, depth_gt_ms, mask_ms, dlossw, focal=False, gamma=0.0, inverse_depth=True):
    depth_loss_weights = dlossw

    loss_dict = {}
    for (stage_inputs, stage_key) in [(inputs[k], k) for k in ['stage1', 'stage2', 'stage3', 'stage4']]:
        depth_gt = depth_gt_ms[stage_key]
        depth_values = stage_inputs["depth_values"]  # (b, d, h, w)
        prob_volume_pre = stage_inputs["prob_volume_pre"].to(torch.float32)
        mask = mask_ms[stage_key]
        mask = (mask > 0.5).to(torch.float32)

        depth_gt = depth_gt.unsqueeze(1)
        depth_gt_volume = depth_gt.expand_as(depth_values)  # (b, d, h, w)
        # inverse depth, depth从大到小变为从小到大
        if inverse_depth:
            depth_values = torch.flip(depth_values, dims=[1])
            prob_volume_pre = torch.flip(prob_volume_pre, dims=[1])
        intervals = torch.abs(depth_values[:, 1:] - depth_values[:, :-1]) / 2  # [b,d-1,h,w]
        intervals = torch.cat([intervals, intervals[:, -1:]], dim=1)  # [b,d,h,w]
        min_depth_values = depth_values[:, 0:1] - intervals[:, 0:1, ]
        max_depth_values = depth_values[:, -1:] + intervals[:, -1:]
        depth_values_right = depth_values + intervals
        out_of_range_left = (depth_gt < min_depth_values).to(torch.float32)
        out_of_range_right = (depth_gt > max_depth_values).to(torch.float32)
        out_of_range_mask = torch.clamp(out_of_range_left + out_of_range_right, 0, 1)
        in_range_mask = 1 - out_of_range_mask
        final_mask = in_range_mask.squeeze(1) * mask
        gt_index_volume = (depth_values_right <= depth_gt_volume).to(torch.float32).sum(dim=1, keepdims=True).to(torch.long)
        gt_index_volume = torch.clamp_max(gt_index_volume, max=depth_values.shape[1] - 1).squeeze(1)

        # mask:[B,H,W], prob:[B,D,H,W], gtd:[B,H,W]
        if focal:
            depth_loss = focal_loss(prob_volume_pre, gt_index_volume, gamma=gamma)
        else:
            final_mask = final_mask.to(torch.bool)
            gt_index_volume = gt_index_volume[final_mask]  # [N,]
            prob_volume_pre = prob_volume_pre.permute(0, 2, 3, 1)[final_mask, :]  # [B,H,W,D]->[N,D]
            depth_loss = F.cross_entropy(prob_volume_pre, gt_index_volume, reduction='mean')
        # depth_loss = torch.sum(depth_loss * final_mask) / (torch.sum(final_mask) + 1e-6)

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            loss_dict[stage_key] = depth_loss_weights[stage_idx] * depth_loss
        else:
            loss_dict[stage_key] = depth_loss

    return loss_dict


def mixup_ce_loss_stage4(inputs, depth_gt_ms, mask_ms, dlossw, inverse_depth=True):
    depth_loss_weights = dlossw

    loss_dict = {}
    for (stage_inputs, stage_key) in [(inputs[k], k) for k in ['stage1', 'stage2', 'stage3', 'stage4']]:
        depth_gt = depth_gt_ms[stage_key]
        depth_values = stage_inputs["depth_values"]  # (b, d, h, w)
        prob_volume_pre = stage_inputs["prob_volume_pre"].to(torch.float32)
        mask = mask_ms[stage_key]
        mask = (mask > 0.5).to(torch.float32)

        depth_gt = depth_gt.unsqueeze(1)
        # inverse depth, depth从大到小变为从小到大
        if inverse_depth:
            depth_values = torch.flip(depth_values, dims=[1])
            prob_volume_pre = torch.flip(prob_volume_pre, dims=[1])

        # 判断out of range
        min_depth_values = depth_values[:, 0:1]
        max_depth_values = depth_values[:, -1:]
        out_of_range_left = (depth_gt < min_depth_values).to(torch.float32)
        out_of_range_right = (depth_gt > max_depth_values).to(torch.float32)
        out_of_range_mask = torch.clamp(out_of_range_left + out_of_range_right, 0, 1)
        in_range_mask = 1 - out_of_range_mask
        final_mask = in_range_mask.squeeze(1) * mask  # [b,h,w]

        # 构建GT index,这里获取的label是0~d-2，左右共用此label
        # |○| | |; label=0 and 1
        # | |○| |; label=1 and 2
        # | | |○|; label=2 and 3
        depth_gt_volume = depth_gt.expand_as(depth_values[:, :-1])  # [b,d-1,h,w]
        gt_index_volume = (depth_values[:, 1:] <= depth_gt_volume).to(torch.float32).sum(dim=1, keepdims=True).to(torch.long)  # [b,1,h,w]
        gt_index_volume = torch.clamp_max(gt_index_volume, max=depth_values.shape[1] - 2).squeeze(1)  # [b,h,w]

        # 构建mix weights，inverse depth的interval当做线性
        gt_depth_left = torch.gather(depth_values[:, :-1], dim=1, index=gt_index_volume.unsqueeze(1))  # [b,1,h,w]
        intervals = torch.abs(depth_values[:, 1:] - depth_values[:, :-1])  # [b,d-1,h,w]
        intervals = torch.gather(intervals, dim=1, index=gt_index_volume.unsqueeze(1))  # [b,1,h,w]
        mix_weights_left = torch.clamp(torch.abs(depth_gt - gt_depth_left) / intervals, 0, 1).squeeze(1)  # [b,1,h,w]->[b,h,w]
        mix_weights_right = 1 - mix_weights_left

        # mask:[B,H,W], prob:[B,D,H,W], gtd:[B,H,W]
        # 分别计算左和右loss
        depth_loss_left = F.cross_entropy(prob_volume_pre[:, :-1], gt_index_volume, reduction='none')  # [b,h,w]
        depth_loss_left = torch.sum(depth_loss_left * mix_weights_left * final_mask) / (torch.sum(final_mask) + 1e-6)
        depth_loss_right = F.cross_entropy(prob_volume_pre[:, 1:], gt_index_volume, reduction='none')  # [b,h,w]
        depth_loss_right = torch.sum(depth_loss_right * mix_weights_right * final_mask) / (torch.sum(final_mask) + 1e-6)
        depth_loss = depth_loss_left + depth_loss_right

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            loss_dict[stage_key] = depth_loss_weights[stage_idx] * depth_loss
        else:
            loss_dict[stage_key] = depth_loss

    return loss_dict


def sigmoid(x, base=2.71828):
    return 1 / (1 + torch.pow(base, -x))
