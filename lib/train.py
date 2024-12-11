import torch
pool = torch.multiprocessing.Pool(torch.multiprocessing.cpu_count(), maxtasksperchild=1)
import numpy as np
import os, sys
from utils.utils import AverageMeter
from utils.eval import calc_all_errs, Evaluation
from utils.img import im_norm_255, im_norm_255_real, vis_err, generate_new_msk
import cv2
from progress.bar import Bar
import os
import utils.fancy_logger as logger
import time
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

np.set_printoptions(threshold = np.inf)

def train(epoch, cfg, data_loader, model, criterions, optimizer=None):
    model.train()
    preds = {}
    Loss = AverageMeter()
    Loss_rot = AverageMeter()
    Loss_rot_coor = AverageMeter()
    Loss_rot_conf = AverageMeter()
    Loss_trans = AverageMeter()
    num_iters = len(data_loader)
    bar = Bar('{}'.format(cfg.pytorch.exp_id[-60:]), max=num_iters)

    time_monitor = False
    vis_dir = os.path.join(cfg.pytorch.save_path, 'train_vis_{}'.format(epoch))
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    for i, (obj, obj_id, inp, target, loss_msk, trans_local, pose, c_box, s_box, box) in enumerate(data_loader):
        cur_iter = i + (epoch - 1) * num_iters
        if cfg.pytorch.gpu > -1:
            inp_var = inp.cuda(cfg.pytorch.gpu, async=True).float()
            target_var = target.cuda(cfg.pytorch.gpu, async=True).float()
            loss_msk_var  = loss_msk.cuda(cfg.pytorch.gpu, async = True).float()
            trans_local_var = trans_local.cuda(cfg.pytorch.gpu, async=True).float()
            pose_var = pose.cuda(cfg.pytorch.gpu, async=True).float()
            c_box_var = c_box.cuda(cfg.pytorch.gpu, async=True).float()
            s_box_var = s_box.cuda(cfg.pytorch.gpu, async=True).float()
        else:
            inp_var = inp.float()
            target_var = target.float()
            loss_msk_var = loss_msk.float()
            trans_local_var = trans_local.float()
            pose_var = pose.float()
            c_box_var = c_box.float()
            s_box_var = s_box.float()

        bs = len(inp)
        # forward propagation
        T_begin = time.time()
        # import ipdb; ipdb.set_trace()
        pred_rot = model(inp_var)
        T_end = time.time() - T_begin
        if time_monitor:
            logger.info("time for a batch forward of resnet model is {}".format(T_end))

        if i % cfg.test.disp_interval == 0:
            # input image
            inp_rgb = (inp[0].cpu().numpy().copy() * 255)[::-1, :, :].astype(np.uint8)
            cfg.writer.add_image('input_image', inp_rgb, i)
            cv2.imwrite(os.path.join(vis_dir, '{}_inp.png'.format(i)), inp_rgb.transpose(1,2,0)[:, :, ::-1])
            if 'rot' in cfg.pytorch.task.lower():
                if not cfg.train.err_res:
                    # coordinates map
                    pred_coor = pred_rot[0, 0:3].data.cpu().numpy().copy()
                    pred_coor[0] = im_norm_255(pred_coor[0])
                    pred_coor[1] = im_norm_255(pred_coor[1])
                    pred_coor[2] = im_norm_255(pred_coor[2])
                    pred_coor = np.asarray(pred_coor, dtype=np.uint8)
                    cv2.imwrite(os.path.join(vis_dir, '{}_coor_x_pred.png'.format(i)), pred_coor[0])
                    cv2.imwrite(os.path.join(vis_dir, '{}_coor_y_pred.png'.format(i)), pred_coor[1])
                    cv2.imwrite(os.path.join(vis_dir, '{}_coor_z_pred.png'.format(i)), pred_coor[2])
                    gt_coor = target[0, 0:3].data.cpu().numpy().copy()
                    gt_coor[0] = im_norm_255(gt_coor[0])
                    gt_coor[1] = im_norm_255(gt_coor[1])
                    gt_coor[2] = im_norm_255(gt_coor[2])
                    gt_coor = np.asarray(gt_coor, dtype=np.uint8)
                    cv2.imwrite(os.path.join(vis_dir, '{}_coor_x_gt.png'.format(i)), gt_coor[0])
                    cv2.imwrite(os.path.join(vis_dir, '{}_coor_y_gt.png'.format(i)), gt_coor[1])
                    cv2.imwrite(os.path.join(vis_dir, '{}_coor_z_gt.png'.format(i)), gt_coor[2])
                    # confidence map
                    pred_conf = pred_rot[0, 3].data.cpu().numpy().copy()
                    pred_conf = (im_norm_255(pred_conf)).astype(np.uint8)
                    cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred.png'.format(i)), pred_conf)
                    gt_conf = target[0, 3].data.cpu().numpy().copy()
                    gt_conf = (im_norm_255(gt_conf)).astype(np.uint8)
                    cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt.png'.format(i)), gt_conf)
                    if cfg.train.split_num > 1:
                        pred_conf_x1 = pred_rot[0, 4].data.cpu().numpy().copy()
                        pred_conf_x1 = (im_norm_255(pred_conf_x1)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_x1.png'.format(i)), pred_conf_x1)
                        gt_conf_x1 = target[0, 4].data.cpu().numpy().copy()
                        gt_conf_x1 = (im_norm_255(gt_conf_x1)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_x1.png'.format(i)), gt_conf_x1)
                        pred_conf_y1 = pred_rot[0, 5].data.cpu().numpy().copy()
                        pred_conf_y1 = (im_norm_255(pred_conf_y1)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_y1.png'.format(i)), pred_conf_y1)
                        gt_conf_y1 = target[0, 5].data.cpu().numpy().copy()
                        gt_conf_y1 = (im_norm_255(gt_conf_y1)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_y1.png'.format(i)), gt_conf_y1)
                        pred_conf_z1 = pred_rot[0, 6].data.cpu().numpy().copy()
                        pred_conf_z1 = (im_norm_255(pred_conf_z1)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_z1.png'.format(i)), pred_conf_z1)
                        gt_conf_z1 = target[0, 6].data.cpu().numpy().copy()
                        gt_conf_z1 = (im_norm_255(gt_conf_z1)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_z1.png'.format(i)), gt_conf_z1)
                    if cfg.train.split_num > 2:
                        pred_conf_x2 = pred_rot[0, 7].data.cpu().numpy().copy()
                        pred_conf_x2 = (im_norm_255(pred_conf_x2)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_x2.png'.format(i)), pred_conf_x2)
                        gt_conf_x2 = target[0, 7].data.cpu().numpy().copy()
                        gt_conf_x2 = (im_norm_255(gt_conf_x2)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_x2.png'.format(i)), gt_conf_x2)
                        pred_conf_y2 = pred_rot[0, 8].data.cpu().numpy().copy()
                        pred_conf_y2 = (im_norm_255(pred_conf_y2)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_y2.png'.format(i)), pred_conf_y2)
                        gt_conf_y2 = target[0, 8].data.cpu().numpy().copy()
                        gt_conf_y2 = (im_norm_255(gt_conf_y2)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_y2.png'.format(i)), gt_conf_y2)
                        pred_conf_z2 = pred_rot[0, 9].data.cpu().numpy().copy()
                        pred_conf_z2 = (im_norm_255(pred_conf_z2)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_z2.png'.format(i)), pred_conf_z2)
                        gt_conf_z2 = target[0, 9].data.cpu().numpy().copy()
                        gt_conf_z2 = (im_norm_255(gt_conf_z2)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_z2.png'.format(i)), gt_conf_z2)
                    if cfg.train.split_num > 4:
                        pred_conf_x3 = pred_rot[0, 10].data.cpu().numpy().copy()
                        pred_conf_x3 = (im_norm_255(pred_conf_x3)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_x3.png'.format(i)), pred_conf_x3)
                        gt_conf_x3 = target[0, 10].data.cpu().numpy().copy()
                        gt_conf_x3 = (im_norm_255(gt_conf_x3)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_x3.png'.format(i)), gt_conf_x3)
                        pred_conf_y3 = pred_rot[0, 11].data.cpu().numpy().copy()
                        pred_conf_y3 = (im_norm_255(pred_conf_y3)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_y3.png'.format(i)), pred_conf_y3)
                        gt_conf_y3 = target[0, 11].data.cpu().numpy().copy()
                        gt_conf_y3 = (im_norm_255(gt_conf_y3)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_y3.png'.format(i)), gt_conf_y3)
                        pred_conf_z3 = pred_rot[0, 12].data.cpu().numpy().copy()
                        pred_conf_z3 = (im_norm_255(pred_conf_z3)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_z3.png'.format(i)), pred_conf_z3)
                        gt_conf_z3 = target[0, 12].data.cpu().numpy().copy()
                        gt_conf_z3 = (im_norm_255(gt_conf_z3)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_z3.png'.format(i)), gt_conf_z3)
                else:
                    # coordinates map
                    pred_coor = pred_rot[0, 0:3].data.cpu().numpy().copy()
                    gt_coor = target[0, 0:3].data.cpu().numpy().copy()
                    err_coor = pred_coor - gt_coor  # err_coor.shape:(3, 64, 64)
                    err_coor = np.abs(err_coor)
                    gt_coor = gt_coor.transpose(1, 2, 0)
                    err_coor = err_coor.transpose(1, 2, 0)
                    gt_coor_vis = np.zeros(gt_coor.shape).astype(np.uint8)
                    gt_coor_vis[..., 0] = im_norm_255_real(gt_coor[..., 0])
                    gt_coor_vis[..., 1] = im_norm_255_real(gt_coor[..., 1])
                    gt_coor_vis[..., 2] = im_norm_255_real(gt_coor[..., 2])
                    cv2.imwrite(os.path.join(vis_dir, '{}_coor_vis.png'.format(i)), gt_coor_vis)
                    gt_conf = target[0, 3].data.cpu().numpy().copy()
                    gt_conf = (im_norm_255(gt_conf)).astype(np.uint8)
                    cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt.png'.format(i)), gt_conf)
                    new_msk = generate_new_msk(gt_coor_vis, gt_conf)
                    cv2.imwrite(os.path.join(vis_dir, '{}_new_msk.png'.format(i)), new_msk)
                    err_coor_vis = vis_err(err_coor, gt_coor_vis, gt_coor)
                    cv2.imwrite(os.path.join(vis_dir, '{}_err_coor_vis.png'.format(i)), err_coor_vis)
                    # confidence map
                    pred_conf = pred_rot[0, 3].data.cpu().numpy().copy()
                    pred_conf = (im_norm_255(pred_conf)).astype(np.uint8)
                    cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred.png'.format(i)), pred_conf)

                    if cfg.train.split_num > 1:
                        pred_conf_x1 = pred_rot[0, 4].data.cpu().numpy().copy()
                        pred_conf_x1 = (im_norm_255(pred_conf_x1)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_x1.png'.format(i)), pred_conf_x1)
                        gt_conf_x1 = target[0, 4].data.cpu().numpy().copy()
                        gt_conf_x1 = (im_norm_255(gt_conf_x1)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_x1.png'.format(i)), gt_conf_x1)
                        pred_conf_y1 = pred_rot[0, 5].data.cpu().numpy().copy()
                        pred_conf_y1 = (im_norm_255(pred_conf_y1)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_y1.png'.format(i)), pred_conf_y1)
                        gt_conf_y1 = target[0, 5].data.cpu().numpy().copy()
                        gt_conf_y1 = (im_norm_255(gt_conf_y1)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_y1.png'.format(i)), gt_conf_y1)
                        pred_conf_z1 = pred_rot[0, 6].data.cpu().numpy().copy()
                        pred_conf_z1 = (im_norm_255(pred_conf_z1)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_z1.png'.format(i)), pred_conf_z1)
                        gt_conf_z1 = target[0, 6].data.cpu().numpy().copy()
                        gt_conf_z1 = (im_norm_255(gt_conf_z1)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_z1.png'.format(i)), gt_conf_z1)
                    if cfg.train.split_num > 2:
                        pred_conf_x2 = pred_rot[0, 7].data.cpu().numpy().copy()
                        pred_conf_x2 = (im_norm_255(pred_conf_x2)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_x2.png'.format(i)), pred_conf_x2)
                        gt_conf_x2 = target[0, 7].data.cpu().numpy().copy()
                        gt_conf_x2 = (im_norm_255(gt_conf_x2)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_x2.png'.format(i)), gt_conf_x2)
                        pred_conf_y2 = pred_rot[0, 8].data.cpu().numpy().copy()
                        pred_conf_y2 = (im_norm_255(pred_conf_y2)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_y2.png'.format(i)), pred_conf_y2)
                        gt_conf_y2 = target[0, 8].data.cpu().numpy().copy()
                        gt_conf_y2 = (im_norm_255(gt_conf_y2)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_y2.png'.format(i)), gt_conf_y2)
                        pred_conf_z2 = pred_rot[0, 9].data.cpu().numpy().copy()
                        pred_conf_z2 = (im_norm_255(pred_conf_z2)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_z2.png'.format(i)), pred_conf_z2)
                        gt_conf_z2 = target[0, 9].data.cpu().numpy().copy()
                        gt_conf_z2 = (im_norm_255(gt_conf_z2)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_z2.png'.format(i)), gt_conf_z2)
                    if cfg.train.split_num > 4:
                        pred_conf_x3 = pred_rot[0, 10].data.cpu().numpy().copy()
                        pred_conf_x3 = (im_norm_255(pred_conf_x3)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_x3.png'.format(i)), pred_conf_x3)
                        gt_conf_x3 = target[0, 10].data.cpu().numpy().copy()
                        gt_conf_x3 = (im_norm_255(gt_conf_x3)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_x3.png'.format(i)), gt_conf_x3)
                        pred_conf_y3 = pred_rot[0, 11].data.cpu().numpy().copy()
                        pred_conf_y3 = (im_norm_255(pred_conf_y3)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_y3.png'.format(i)), pred_conf_y3)
                        gt_conf_y3 = target[0, 11].data.cpu().numpy().copy()
                        gt_conf_y3 = (im_norm_255(gt_conf_y3)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_y3.png'.format(i)), gt_conf_y3)
                        pred_conf_z3 = pred_rot[0, 12].data.cpu().numpy().copy()
                        pred_conf_z3 = (im_norm_255(pred_conf_z3)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred_z3.png'.format(i)), pred_conf_z3)
                        gt_conf_z3 = target[0, 12].data.cpu().numpy().copy()
                        gt_conf_z3 = (im_norm_255(gt_conf_z3)).astype(np.uint8)
                        cv2.imwrite(os.path.join(vis_dir, '{}_conf_gt_z3.png'.format(i)), gt_conf_z3)

        # loss
        if 'rot' in cfg.pytorch.task.lower() and not cfg.network.rot_head_freeze:
            if cfg.loss.rot_mask_loss:
                loss_rot_coor = criterions[cfg.loss.rot_loss_type](loss_msk_var[:3] * pred_rot[:3], loss_msk_var[:3] * target_var[:3])
                loss_rot_conf = criterions[cfg.loss.rot_loss_type](loss_msk_var[3:] * pred_rot[3:], loss_msk_var[3:] * target_var[3:])
                loss_rot = loss_rot_coor + loss_rot_conf
            else:
                loss_rot = criterions[cfg.loss.rot_loss_type](pred_rot, target_var)
        else:
            loss_rot = 0

        Loss_rot.update(loss_rot.item() if loss_rot != 0 else 0, bs)
        Loss_rot_coor.update(loss_rot_coor.item() if loss_rot_coor != 0 else 0, bs)
        Loss_rot_conf.update(loss_rot_conf.item() if loss_rot_conf != 0 else 0, bs)

        cfg.writer.add_scalar('data/loss_rot', loss_rot.item() if loss_rot != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/loss_rot_coor', loss_rot_coor.item() if loss_rot_coor != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/loss_rot_conf', loss_rot_conf.item() if loss_rot_conf != 0 else 0, cur_iter)

        optimizer.zero_grad()
        model.zero_grad()
        T_begin = time.time()
        loss_rot.backward()
        optimizer.step()
        T_end = time.time() - T_begin
        if time_monitor:
            logger.info("time for backward of model: {}".format(T_end))
       
        Bar.suffix = 'train Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss_rot {loss_rot.avg:.4f} | Loss_rot_coor {loss_rot_coor.avg:.4f} | Loss_rot_conf {loss_rot_conf.avg:.4f}'.format(
            epoch, i, num_iters, total=bar.elapsed_td, eta=bar.eta_td, loss_rot=Loss_rot, loss_rot_coor=Loss_rot_coor, loss_rot_conf=Loss_rot_conf)
        bar.next()
    bar.finish()
    return {'Loss_rot': Loss_rot.avg, 'Loss_rot_coor': Loss_rot_coor.avg, 'Loss_rot_conf': Loss_rot_conf.avg}, preds
