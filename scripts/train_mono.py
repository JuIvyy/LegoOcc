import os, time, argparse, os.path as osp, numpy as np
import sys
import torch
import gc
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

from legoocc.occutils.iou_eval import IOUEvalBatch
from legoocc.occutils.iou_as_iso import SSCMetrics, SSCMetricsTorch
from legoocc.occutils.loss_record import LossRecord
from legoocc.occutils.load_save_util import revise_ckpt, revise_ckpt_2
from legoocc.loss.depth_loss import DepthLoss

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim.optimizer.builder import build_optim_wrapper
from mmengine.logging.logger import MMLogger
from mmengine.utils import symlink
from timm.scheduler import CosineLRScheduler
import open3d as o3d
import cv2
from einops import einsum, rearrange
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
import pickle

import warnings
warnings.filterwarnings("ignore")

from PIL import Image

from legoocc.model.encoder.gaussianformer.utils import GaussianPrediction


def pass_print(*args, **kwargs):
    pass


def is_main_process():
    if not dist.is_available():
        return True
    elif not dist.is_initialized():
        return True
    else:
        return dist.get_rank() == 0


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def save_model(model, optimizer=None, scheduler=None, epoch=None, **kwargs):
    dict_to_save = {
        'state_dict': model.state_dict()
    }
    if optimizer is not None:
        dict_to_save['optimizer'] = optimizer.state_dict()

    if scheduler is not None:
        dict_to_save['scheduler'] = scheduler.state_dict()

    if len(kwargs):
        dict_to_save.update(kwargs)

    if epoch is not None:
        dict_to_save['epoch'] = epoch
        save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch}.pth')
    else:
        save_file_name = os.path.join(os.path.abspath(args.work_dir), f'ckpt.pth')

    torch.save(dict_to_save, save_file_name)
    dst_file = osp.join(args.work_dir, 'latest.pth')
    symlink(save_file_name, dst_file)


def main(args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    set_random_seed(cfg.seed)
    cfg.work_dir = args.work_dir
    max_num_epochs = cfg.max_epochs
    eval_freq = cfg.eval_freq
    print_freq = cfg.print_freq
    vis_freq = cfg.get('vis_freq', 10)
    if args.vis_freq:
        vis_freq = args.vis_freq

    # init DDP
    distributed = True
    world_size = int(os.environ["WORLD_SIZE"])  # number of nodes
    rank = int(os.environ["RANK"])  # node id
    # gpu = int(os.environ['LOCAL_RANK'])

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend="nccl",
        # init_method=f"env://", 
        world_size=world_size,
    )
    rank, world_size = get_dist_info()

    if not is_main_process():
        import builtins
        builtins.print = pass_print

    # configure logger
    if is_main_process():
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    exp_name = os.path.basename(args.work_dir)
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger(
        name='embodied',
        logger_name=exp_name,
        log_file=log_file,
        log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')
    # dist.barrier()

    # build model
    from legoocc.model import build_model
    my_model = build_model(cfg.model).cuda()
    
    if cfg.flag_depthanything_as_gt:
        my_model.depthanything.requires_grad_(False)

    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    # logger.info(f'Model:\n{my_model}')

    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        if cfg.get('track_running_stats', False):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model,
            # device_ids=[gpu],
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)

    logger.info('done ddp model')

    # build dataloader
    from legoocc.dataset import build_dataloader
    train_dataset_loader, val_dataset_loader = \
        build_dataloader(
            cfg.train_dataset_config,
            cfg.val_dataset_config,
            cfg.train_wrapper_config,
            cfg.val_wrapper_config,
            cfg.train_loader_config,
            cfg.val_loader_config,
            dist=distributed,
        )

    # get optimizer, loss, scheduler
    amp = cfg.get('amp', True)
    optimizer = build_optim_wrapper(my_model, cfg.optimizer_wrapper)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    from legoocc.loss import GPD_LOSS
    loss_func = GPD_LOSS.build(cfg.loss).cuda() if cfg.loss is not None else None
    occ_loss_func = GPD_LOSS.build(cfg.occ_loss).cuda() if cfg.occ_loss is not None else None
    loss_depth_func = GPD_LOSS.build(cfg.depth_loss).cuda()
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataset_loader)*max_num_epochs,
        lr_min=1e-6,
        warmup_t=1000, # FIXME
        warmup_lr_init=1e-6,
        t_in_epochs=False
    )

    # CalMeanIou = SSCMetrics(n_classes=12)
    CalMeanIou = SSCMetricsTorch(n_classes=12)
    CalMeanIouOpen = SSCMetricsTorch(n_classes=12)
    # resume and load
    epoch = 0
    best_val_iou = 0
    best_val_miou = 0
    global_iter = 0

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    logger.info(f'resume from: {cfg.resume_from}')
    logger.info(f'work dir: {args.work_dir}')

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        if 'best_val_iou' in ckpt:
            best_val_iou = ckpt['best_val_iou']
        if 'best_val_miou' in ckpt:
            best_val_miou = ckpt['best_val_miou']
        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:

            state_dict = revise_ckpt_2(state_dict)
            keys_to_skip = []
            for k in list(state_dict.keys()):
                if k in my_model.state_dict():
                    model_shape = my_model.state_dict()[k].shape
                    ckpt_shape = state_dict[k].shape
                    if model_shape != ckpt_shape:
                        print(f"Skip loading '{k}' due to shape mismatch: model {model_shape}, ckpt {ckpt_shape}")
                        keys_to_skip.append(k)
            for k in keys_to_skip:
                state_dict.pop(k)

            print(my_model.load_state_dict(state_dict, strict=False))

    metas_tensor_keys_inv = ['depth_gt_np_valid', 'depth_gt_np', 'name', 'cam2img', 'world2img', 'rgb_path', 'depth_path','num_depth', 'occ_mask_valid', 'occ_mask_valid_fov', 'img_shape', 'img_aug_matrix']

    total_params = sum(p.numel() for p in my_model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f} M")

    if args.evaluate:
        my_model.eval()
        CalMeanIou.reset()
        CalMeanIouOpen.reset()
        # loss_record = LossRecord(loss_func=loss_func)
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        with torch.no_grad():
            for i_iter_val, data in enumerate(tqdm(val_dataset_loader)):

                for i in range(len(data)):
                    if isinstance(data[i], torch.Tensor):
                        data[i] = data[i].cuda()
                (imgs, metas, label) = data

                with torch.cuda.amp.autocast(enabled=amp, dtype=torch.bfloat16):

                    prompt = None
                    # classnames = result_dict['classnames']
                    classnames = my_model.module.ov_model.query_words

                    result_dict, my_occ, predtoreturn = my_model(
                        imgs=imgs,
                        metas=metas,
                        points=None,
                        label=label,
                        grad_frames=None,
                        test_mode=True,
                        test_prompt=prompt,
                    )

                if my_model.module.semantic_dim == 0:
                    occ_ov_cos_sim = result_dict['occ_ov_cos_sim']
                    occ_ov_pred = torch.argmax(occ_ov_cos_sim, dim=1)
                    # add 1 for ignore class 0
                    occ_ov_pred = occ_ov_pred + 1
                    empty_pred = result_dict['ce_input'].squeeze(1)
                    occ_ov_pred[empty_pred > 0.5] = 12
                    voxel_predict = occ_ov_pred
                else:
                    voxel_predict = result_dict['ce_input'].argmax(dim=1).long()

                voxel_label = result_dict['ce_label'].long()

                voxel_predict[voxel_predict == 0] = 255
                voxel_predict[voxel_predict == 12] = 0
                voxel_label[voxel_label == 0] = 255
                voxel_label[voxel_label == 12] = 0

                # voxel_predict = voxel_predict.cpu()
                # voxel_label = voxel_label.cpu()
                CalMeanIou.add_batch(voxel_predict, voxel_label)

                neg_occ_mask = voxel_predict == 0
                occ_ov_cos_sim = rearrange(result_dict['occ_ov_cos_sim'], 'b c h w d -> b h w d c')

                max_value = 1.1
                min_value = -1.1

                occ_ov_logit = F.pad(occ_ov_cos_sim, (1, 0), value=min_value)  # [..., C+1]
                occ_ov_logit[torch.nonzero(neg_occ_mask, as_tuple=True) + (0,)] = max_value
                occ_ov_score, occ_ov_pred = torch.max(occ_ov_logit, dim=-1)    # [...], [...]

                CalMeanIouOpen.add_batch(occ_ov_pred, voxel_label)

                if i_iter_val % print_freq == 0:
                    stats = CalMeanIou.get_stats(distributed=False)
                    info_sem_cls = stats["iou_ssc"]
                    info_sem = stats["iou_ssc_mean"]
                    info_geo = stats["iou"]

                    ov_stats = CalMeanIouOpen.get_stats(distributed=False)
                    ov_info_sem_cls = ov_stats["iou_ssc"]
                    ov_info_sem = ov_stats["iou_ssc_mean"]
                    ov_info_geo = ov_stats["iou"]

                    if is_main_process():
                        logger.info(f'Current val iou of sem_cls is {info_sem_cls}')
                        logger.info(f'Current val iou of sem is {info_sem}')
                        logger.info(f'Current val iou of geo is {info_geo}')

                        logger.info(f'Current val OPEN iou of sem_cls is {ov_info_sem_cls}')
                        logger.info(f'Current val OPEN iou of sem is {ov_info_sem}')
                        logger.info(f'Current val OPEN iou of geo is {ov_info_geo}')

                gc.collect()
                torch.cuda.empty_cache()

            stats = CalMeanIou.get_stats(distributed=distributed)
            open_stats = CalMeanIouOpen.get_stats(distributed=distributed)

            info_sem_cls = stats["iou_ssc"]
            info_sem = stats["iou_ssc_mean"]
            info_geo = stats["iou"]
            
            ov_info_sem_cls = open_stats["iou_ssc"]
            ov_info_sem = open_stats["iou_ssc_mean"]
            ov_info_geo = open_stats["iou"]

            local_vals = torch.stack(num_gaussians).float() if isinstance(num_gaussians[0], torch.Tensor) else torch.tensor(num_gaussians, dtype=torch.float32, device='cuda')
            local_sum = local_vals.sum()
            local_cnt = torch.tensor([len(local_vals)], dtype=torch.long, device=local_sum.device)
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_cnt, op=dist.ReduceOp.SUM)
            global_cnt = max(int(local_cnt.item()), 1)
            global_mean = (local_sum / local_cnt.clamp_min(1)).item()

            if is_main_process():
                logger.info(f'Current val iou of sem_cls is {info_sem_cls}')
                logger.info(f'Current val iou of sem is {info_sem}')
                logger.info(f'Current val iou of geo is {info_geo}')

                logger.info(f'Current val OPEN iou of sem_cls is {ov_info_sem_cls}')
                logger.info(f'Current val OPEN iou of sem is {ov_info_sem}')
                logger.info(f'Current val OPEN iou of geo is {ov_info_geo}')

                logger.info(f'Current avg number of gaussian is {global_mean}')

        return

    # training
    while epoch < max_num_epochs:

        my_model.train()
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_record = LossRecord(loss_func=loss_func)
        time.sleep(1)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, data in enumerate(train_dataset_loader):
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].cuda()
            (imgs, metas, label) = data

            # Calculate global training ratio: 0 at beginning, 1 at the end
            global_training_ratio = (epoch * len(train_dataset_loader) + i_iter) / (max_num_epochs * len(train_dataset_loader))

            data_time_e = time.time()

            with torch.cuda.amp.autocast(enabled=amp):
                result_dict, sparse_result_dict, predtoreturn = my_model(
                    imgs=imgs, metas=metas,
                    points=None, label=label,
                    grad_frames=cfg.grad_frames,
                    global_training_ratio=global_training_ratio,
                    test_mode=False)

            assert cfg.ignore_label == 0

            if my_model.module.semantic_dim == 0 and 'ce_label' in result_dict:
                labels_nosem = result_dict['ce_label'].clone().long()
                labels_nosem[torch.logical_and(labels_nosem > 0, labels_nosem < 12)] = 1
                labels_nosem[labels_nosem == 12] = 2
                # occ_label = (result_dict['ce_label'] == cfg.empty_idx).long() + 1
                # occ_label[result_dict['ce_label'] == 0] = 0
                # result_dict['occ_label'] = occ_label
                occ_pred = result_dict['ce_input']
                occ_pred = torch.cat([
                    torch.zeros_like(occ_pred),
                    1 - occ_pred, occ_pred,
                ], dim=1)
                result_dict.update(dict(
                    ce_input=occ_pred,
                    ce_label=labels_nosem
                ))

            total_loss = 0.
            all_loss_dict = {}

            if loss_depth_func.weight > 0:
                depth_loss = loss_depth_func(result_dict)
                if isinstance(depth_loss, torch.Tensor):
                    total_loss = total_loss + depth_loss
                    all_loss_dict['depth_loss'] = depth_loss.detach().item()
                else:
                    depth_loss, depth_loss_dict = depth_loss
                    total_loss = total_loss + depth_loss
                    all_loss_dict.update(depth_loss_dict)

            if 'ce_label' in result_dict:
                occ_loss, occ_loss_dict = occ_loss_func(result_dict)
                all_loss_dict.update(occ_loss_dict)
                total_loss = total_loss + occ_loss

            if loss_func is not None:
                loss, loss_dict = loss_func(result_dict)
                all_loss_dict.update(loss_dict)
                total_loss = total_loss + loss

            loss_record.update(loss=total_loss.item(), loss_dict=all_loss_dict)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)

            valid_grad = True

            scaler.step(optimizer)
            scaler.update()
            scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0 and is_main_process():
                lr = optimizer.param_groups[0]['lr']
                loss_info = loss_record.loss_info()
                logger.info('[TRAIN] Epoch %d Iter %5d/%d   ' % (epoch+1, i_iter, len(train_dataset_loader)) + loss_info +
                            'GradNorm: %.3f,   lr: %.7f,   time: %.3f (%.3f)' % (grad_norm, lr, time_e - time_s, data_time_e - data_time_s))
                loss_record.reset()
            data_time_s = time.time()
            time_s = time.time()

        if is_main_process():
            save_model(
                my_model, optimizer,
                scheduler, epoch + 1, 
                global_iter=global_iter,
                best_val_iou=best_val_iou,
                best_val_miou=best_val_miou)

        epoch += 1

        # eval
        if epoch % eval_freq == 0:
            my_model.eval()
            CalMeanIou.reset()
            CalMeanIouOpen.reset()
            loss_record = LossRecord(loss_func=loss_func)
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            with torch.no_grad():
                for i_iter_val, data in enumerate(val_dataset_loader):
                    for i in range(len(data)):
                        if isinstance(data[i], torch.Tensor):
                            data[i] = data[i].cuda()
                    (imgs, metas, label) = data

                    with torch.cuda.amp.autocast(enabled=amp):
                        result_dict, my_occ, predtoreturn = my_model(
                            imgs=imgs,
                            metas=metas,
                            points=None,
                            label=label,
                            grad_frames=None,
                            test_mode=True
                        )

                    if my_model.module.semantic_dim == 0:
                        occ_ov_cos_sim = result_dict['occ_ov_cos_sim']
                        occ_ov_pred = torch.argmax(occ_ov_cos_sim, dim=1)
                        # add 1 for ignore class 0
                        occ_ov_pred = occ_ov_pred + 1
                        empty_pred = result_dict['ce_input'].squeeze(1)
                        occ_ov_pred[empty_pred > 0.5] = 12
                        voxel_predict = occ_ov_pred
                    else:
                        voxel_predict = result_dict['ce_input'].argmax(dim=1).long()

                    voxel_label = result_dict['ce_label'].long()
                    
                    voxel_predict[voxel_predict == 0] = 255
                    voxel_predict[voxel_predict == 12] = 0
                    voxel_label[voxel_label == 0] = 255
                    voxel_label[voxel_label == 12] = 0
                    # voxel_predict = voxel_predict.cpu()
                    # voxel_label = voxel_label.cpu()

                    CalMeanIou.add_batch(voxel_predict, voxel_label)

                    neg_occ_mask = voxel_predict == 0
                    occ_ov_cos_sim = rearrange(result_dict['occ_ov_cos_sim'], 'b c h w d -> b h w d c')
                    max_value = 1.1
                    min_value = -1.1
                    occ_ov_logit = F.pad(occ_ov_cos_sim, (1, 0), value=min_value)
                    occ_ov_logit[torch.nonzero(neg_occ_mask, as_tuple=True) + (0,)] = max_value
                    occ_ov_pred = torch.argmax(occ_ov_logit, dim=-1)
                    CalMeanIouOpen.add_batch(occ_ov_pred, voxel_label)

                    gc.collect()
                    torch.cuda.empty_cache()

            stats = CalMeanIou.get_stats(distributed=distributed)
            info_sem_cls = stats["iou_ssc"]
            info_sem = stats["iou_ssc_mean"]
            info_geo = stats["iou"]

            ov_stats = CalMeanIouOpen.get_stats(distributed=False)
            ov_info_sem_cls = ov_stats["iou_ssc"]
            ov_info_sem = ov_stats["iou_ssc_mean"]
            ov_info_geo = ov_stats["iou"]

            logger.info(f'Current val iou of sem_cls is {info_sem_cls}')
            logger.info(f'Current val iou of sem is {info_sem}')
            logger.info(f'Current val iou of geo is {info_geo}')
            logger.info(f'Current val OPEN iou of sem_cls is {ov_info_sem_cls}')
            logger.info(f'Current val OPEN iou of sem is {ov_info_sem}')
            logger.info(f'Current val OPEN iou of geo is {ov_info_geo}')


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/train_mono_config.py')
    parser.add_argument('--work-dir', type=str, default='/home/wyq/WorkSpace/workdir/train_mono')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_freq', type=int, default=10)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--ov_use_vlm', action='store_true')

    args, _ = parser.parse_known_args()
    main(args)
