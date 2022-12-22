import collections
import os
import time

import cv2
import torch.distributed as dist
from tqdm import tqdm

from base import BaseTrainer
from models.losses import *
from utils import *


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, optimizer, config, data_loader, valid_data_loader=None, lr_scheduler=None, writer=None, rank=0, ddp=False,
                 train_sampler=None, debug=False):
        super().__init__(model, optimizer, config, writer=writer, rank=rank, ddp=ddp)
        self.config = config
        self.ddp = ddp
        self.debug = debug
        self.data_loader = data_loader
        self.train_sampler = train_sampler
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = config['trainer']['logging_every']
        self.depth_type = self.config['arch']['args']['depth_type']
        self.ndepths = self.config['arch']['args']['ndepths']
        self.random_mask = self.config['data_loader'][0]['args'].get('random_mask', False)
        self.scale_batch_map = self.config['data_loader'][0]['args']['multi_scale_args']['scale_batch_map']
        self.focal = config['arch']['args'].get('focal', False)
        self.gamma = config['arch']['args'].get('gamma', 0.0)
        self.inverse_depth = config['arch']['args']['inverse_depth']
        self.mask_out_range = config['arch']['args'].get('mask_out_range', False)
        self.grad_norm = config['trainer'].get('grad_norm', None)
        self.train_metrics = DictAverageMeter()
        self.valid_metrics = DictAverageMeter()
        self.scale_dir = True
        if config['fp16'] is True:
            self.fp16 = True
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.fp16 = False

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        # if self.ddp:
        self.train_sampler.set_epoch(epoch)  # Shuffle each epoch
        self.data_loader[0].dataset.reset_dataset(self.train_sampler)

        self.model.train()
        dist_group = torch.distributed.group.WORLD

        # if self.rank == 0:
        #     val_metrics = self._valid_epoch(epoch)

        global_step = 0
        import collections
        scaled_grads = collections.defaultdict(list)

        # training
        for dl in self.data_loader:
            for batch_idx, sample in enumerate(dl):

                start_time = time.time()

                num_stage = 4
                sample_cuda = tocuda(sample)
                depth_gt_ms = sample_cuda["depth"]
                mask_ms = sample_cuda["mask"]
                imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]
                depth_values = sample_cuda["depth_values"]
                depth_interval = depth_values[:, 1] - depth_values[:, 0]

                self.optimizer.zero_grad()

                # gradient accumulate
                bs = self.scale_batch_map[str(imgs.shape[3])]
                iters_to_accumulate = imgs.shape[0] // bs
                total_loss = 0.0
                total_loss_dict = collections.defaultdict(float)

                for bi in range(iters_to_accumulate):
                    b_start = bi * bs
                    b_end = (bi + 1) * bs
                    cam_params_tmp = {}
                    depth_gt_ms_tmp = {}
                    mask_ms_tmp = {}
                    imgs_tmp = imgs[b_start:b_end]
                    for k in cam_params:
                        cam_params_tmp[k] = cam_params[k][b_start:b_end]
                        depth_gt_ms_tmp[k] = depth_gt_ms[k][b_start:b_end]
                        mask_ms_tmp[k] = mask_ms[k][b_start:b_end]

                    if self.fp16:
                        with torch.cuda.amp.autocast():
                            outputs = self.model.forward(imgs_tmp, cam_params_tmp, depth_values[b_start:b_end])
                    else:
                        outputs = self.model.forward(imgs_tmp, cam_params_tmp, depth_values[b_start:b_end])

                    if self.depth_type == 're':
                        loss_dict = reg_loss_stage4(outputs, depth_gt_ms_tmp, mask_ms_tmp,
                                                    dlossw=[1, 1, 1, 1], depth_interval=depth_interval[b_start:b_end],
                                                    mask_out_range=self.mask_out_range, inverse_depth=self.inverse_depth)
                    elif self.depth_type == 'was':
                        loss_dict = wasserstein_loss(outputs, depth_gt_ms_tmp, mask_ms_tmp,
                                                     dlossw=[1, 1, 1, 1], ot_iter=10, ot_eps=1, ot_continous=False,
                                                     inverse=self.config['arch']['args']['inverse_depth'])
                    elif self.depth_type == 'ce':
                        loss_dict = ce_loss_stage4(outputs, depth_gt_ms_tmp, mask_ms_tmp, dlossw=[1, 1, 1, 1],
                                                   focal=self.focal, gamma=self.gamma, inverse_depth=self.inverse_depth)
                    elif self.depth_type == 'mixup_ce':
                        loss_dict = mixup_ce_loss_stage4(outputs, depth_gt_ms_tmp, mask_ms_tmp, dlossw=[1, 1, 1, 1],
                                                         inverse_depth=self.inverse_depth)
                    else:
                        raise NotImplementedError

                    loss = torch.tensor(0.0)
                    for key in loss_dict:
                        loss = loss + loss_dict[key] / iters_to_accumulate
                        total_loss_dict[key] = total_loss_dict[key] + loss_dict[key] / iters_to_accumulate

                    total_loss += loss

                    if self.fp16:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                if self.debug:
                    # DEBUG:scaled grad
                    with torch.no_grad():
                        for group in self.optimizer.param_groups:
                            for param in group["params"]:
                                if param.grad is None:
                                    continue
                                if param.grad.is_sparse:
                                    if param.grad.dtype is torch.float16:
                                        param.grad = param.grad.coalesce()
                                    to_unscale = param.grad._values()
                                else:
                                    to_unscale = param.grad
                                v = to_unscale.clone().abs().max()
                                if torch.isinf(v) or torch.isnan(v):
                                    print('Rank', str(self.rank) + ':', 'INF in', group['layer_name'], 'of step', global_step, '!!!')
                                scaled_grads[group['layer_name']].append(v.item() / self.scaler.get_scale())

                if self.grad_norm is not None:
                    if self.fp16:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm, error_if_nonfinite=False)

                if self.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.lr_scheduler.step()

                # forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1000.0 ** 2)
                # print(imgs.shape, 'max_mem:', forward_max_memory_allocated)
                # print(imgs.shape[3], imgs.shape[4])

                global_step = (epoch - 1) * len(dl) + batch_idx

                if self.debug and batch_idx % 50 == 0 and self.rank == 0:
                    scaled_grads_dict = {}
                    for k in scaled_grads:
                        scaled_grads_dict[k] = np.max(scaled_grads[k])
                    save_scalars(self.writer, 'grads', scaled_grads_dict, global_step)
                    scaled_grads = collections.defaultdict(list)

                if batch_idx % self.log_step == 0 and self.rank == 0:
                    scalar_outputs = {"loss": total_loss.item()}
                    for key in total_loss_dict:
                        scalar_outputs['loss_' + key] = loss_dict[key].item()
                    image_outputs = {"pred_depth": outputs['refined_depth'] * mask_ms_tmp[f'stage{num_stage}'],
                                     "pred_depth_nomask": outputs['refined_depth'], "conf": outputs['photometric_confidence'],
                                     "gt_depth": depth_gt_ms_tmp[f'stage{num_stage}'], "ref_img": imgs_tmp[:, 0]}
                    save_scalars(self.writer, 'train', scalar_outputs, global_step)
                    save_images(self.writer, 'train', image_outputs, global_step)
                    print_str = "Epoch {}/{}, Iter {}/{}, lr={:.1e}, loss={:.2f}, ".format(epoch, self.epochs, batch_idx, len(dl), self.optimizer.param_groups[0]["lr"],
                                                                                           total_loss.item())
                    print_str += "time={:.2f}".format(time.time() - start_time)
                    print_str += ", size:{}x{}, bs:{}".format(imgs.shape[3], imgs.shape[4], bs)
                    if self.fp16:
                        print_str += ', scale={:d}'.format(int(self.scaler.get_scale()))
                    print(print_str)

                    del scalar_outputs, image_outputs

        val_metrics = self._valid_epoch(epoch)
        dist.barrier(group=dist_group)
        for k in val_metrics:
            dist.all_reduce(val_metrics[k], group=dist_group, async_op=False)
            val_metrics[k] /= dist.get_world_size(dist_group)
            val_metrics[k] = val_metrics[k].item()
        if self.rank == 0:
            save_scalars(self.writer, 'test', val_metrics, epoch)
            print("Global Test avg_test_scalars:", val_metrics)
        else:
            val_metrics = {'useless_for_other_ranks': -1}
        dist.barrier(group=dist_group)

        return val_metrics

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        with torch.no_grad():
            for dl in self.valid_data_loader:
                self.valid_metrics.reset()
                for batch_idx, sample in enumerate(tqdm(dl)):
                    sample_cuda = tocuda(sample)
                    depth_gt_ms = sample_cuda["depth"]
                    mask_ms = sample_cuda["mask"]
                    num_stage = 4
                    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
                    mask = mask_ms["stage{}".format(num_stage)]

                    imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]

                    depth_values = sample_cuda["depth_values"]
                    depth_interval = depth_values[:, 1] - depth_values[:, 0]
                    if self.fp16:
                        with torch.cuda.amp.autocast():
                            outputs = self.model.forward(imgs, cam_params, depth_values)
                    else:
                        outputs = self.model.forward(imgs, cam_params, depth_values)

                    depth_est = outputs["refined_depth"].detach()
                    if self.config['data_loader'][0]['type'] == 'BlendedLoader':
                        scalar_outputs = collections.defaultdict(float)
                        for j in range(depth_interval.shape[0]):
                            di = depth_interval[j].item()
                            scalar_outputs_ = {"abs_depth_error": AbsDepthError_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5),
                                               "thres2mm_error": Thres_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, di * 2),
                                               "thres4mm_error": Thres_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, di * 4),
                                               "thres8mm_error": Thres_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, di * 8),
                                               "thres14mm_error": Thres_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, di * 14)}
                            for k in scalar_outputs_:
                                scalar_outputs[k] += scalar_outputs_[k]
                        for k in scalar_outputs:
                            scalar_outputs[k] /= depth_interval.shape[0]
                    else:
                        di = depth_interval[0].item() / 2.65
                        scalar_outputs = {"abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                                          "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di * 2),
                                          "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di * 4),
                                          "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di * 8),
                                          "thres14mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di * 14)}

                    scalar_outputs = tensor2float(scalar_outputs)
                    image_outputs = {"pred_depth": outputs['refined_depth'] * mask, "gt_depth": depth_gt, "ref_img": imgs[:, 0]}

                    self.valid_metrics.update(scalar_outputs)

                if self.rank == 0:
                    save_images(self.writer, 'test', image_outputs, epoch)
                val_metrics = self.valid_metrics.mean()
                val_metrics['mean_error'] = val_metrics['thres2mm_error'] + val_metrics['thres4mm_error'] + val_metrics['thres8mm_error'] + val_metrics['thres14mm_error']
                val_metrics['mean_error'] = val_metrics['mean_error'] / 4.0
                # save_scalars(self.writer, 'test', val_metrics, epoch)
                # print(f"Rank{self.rank}, avg_test_scalars:", val_metrics)

        for k in val_metrics:
            val_metrics[k] = torch.tensor(val_metrics[k], device=self.rank, dtype=torch.float32)
        self.model.train()

        return val_metrics
