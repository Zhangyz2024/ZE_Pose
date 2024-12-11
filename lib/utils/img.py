import numpy as np
import cv2

np.set_printoptions(threshold = np.inf)

def vis_err(err_coor, gt_coor_vis, gt_coor):
    x_err_l = []
    y_err_l = []
    z_err_l = []
    pts = []
    #print('gt_coor', gt_coor)
    #print('err_coor', err_coor)
    for x in range(err_coor.shape[1]):
        for y in range(err_coor.shape[0]):
            if not(gt_coor[y, x, 0] == 127.5 and gt_coor[y, x, 1] == 127.5 and gt_coor[y, x, 2] == 127.5) and \
               not(gt_coor[y, x, 0] == 0.0 and gt_coor[y, x, 1] == 0.0 and gt_coor[y, x, 2] == 0.0):
                x_err_l.append(err_coor[y, x, 0])
                y_err_l.append(err_coor[y, x, 1])
                z_err_l.append(err_coor[y, x, 2])
                pts.append([x, y])
    x_err_l = np.array(x_err_l)
    y_err_l = np.array(y_err_l)
    z_err_l = np.array(z_err_l)
    pts = np.array(pts)
    pts_x = pts[np.argsort(x_err_l)][-int(0.05 * len(pts)):-1]
    pts_y = pts[np.argsort(y_err_l)][-int(0.05 * len(pts)):-1]
    pts_z = pts[np.argsort(z_err_l)][-int(0.05 * len(pts)):-1]
    for pts in pts_x:
        gt_coor_vis[pts[1], pts[0], :] = np.array([0, 0, 0])
    for pts in pts_y:
        gt_coor_vis[pts[1], pts[0], :] = np.array([0, 0, 0])
    for pts in pts_z:
        gt_coor_vis[pts[1], pts[0], :] = np.array([0, 0, 0])
    return gt_coor_vis

def coor_Laplacian(coor):
    w = coor.shape[1]
    h = coor.shape[0]
    coor_diff = np.zeros((h-2,w-2,3)).astype(np.float32)
    for x in range(1, w-1):
        for y in range(1, h-1):
                coor_diff[y-1, x-1, :] = (8 * coor[y, x, :] - coor[y-1, x, :] - coor[y+1, x, :] - coor[y, x-1, :] - coor[y, x+1, :]     \
                                         - coor[y-1, x-1, :] - coor[y-1, x+1, :] - coor[y+1, x-1, :] - coor[y+1, x+1, :]) / 8
    return coor_diff

def visualize_msk(msk, pts_laps):
    vis_msk = np.stack([msk, msk, msk], axis=2)
    for pts_lap in pts_laps:
        vis_msk[pts_lap[1], pts_lap[0], 0] = 255
        vis_msk[pts_lap[1], pts_lap[0], 1] = 0
        vis_msk[pts_lap[1], pts_lap[0], 2] = 255
    return vis_msk

def generate_new_coor_diff_big(coor):
    w = 64
    h = 64
    coor_back = np.zeros((h+2, w+2, 3)).astype(np.float32)
    coor_back[1:h+1, 1:w+1, :] = coor[:, :, :]
    coor_diff = coor_Laplacian(coor_back)
    coor_diff_big = np.zeros((64, 64, 3)).astype(np.float32)
    coor_diff_big[0:h, 0:w, :] = coor_diff[:, :, :]
    return coor_diff_big

def generate_new_msk(coor, msk):
    coor_diff = generate_new_coor_diff_big(coor)
    pts_laps = []
    msk_cache = msk
    out = np.zeros(msk.shape)
    out[:, :] = msk[:, :]
    for x in range(0, 64):
        for y in range(0, 64):
            if msk_cache[y, x] == 0.0:
                continue
            if (np.abs(coor_diff[y,x,0]) > 0.15 * 127 or np.abs(coor_diff[y,x,1]) > 0.15 * 127 or np.abs(coor_diff[y,x,2]) > 0.15 * 127):
                out[y, x] = 0.0
                pts_laps.append([x, y])
    vis_msk = visualize_msk(msk, pts_laps)
    return vis_msk

def im_norm(im):
    """
    normalize to [0, 1].
    """
    if im.max() == im.min():
        #print('all elements identical!!!')
        return np.ones_like(im)
    else:
        return (im - im.min()) / (im.max() - im.min())

def im_norm_255_real(im):
    im[:, :] = (im[:, :] + 1) / 2 * 255
    im.astype(np.uint8)
    return im

def im_norm_255(im):
    """
    normalize to [0, 255].
    """
    if im.max() == im.min():
        #print('all elements identical!!!')
        return np.ones_like(im)
    else:    
        return (im - im.min()) * 255. / (im.max() - im.min())

def xyxy_iou(box1,box2):
    """
    calculate iou between box1 and box2
    :param box1: (4, ), format (left, upper, right, bottom)
    :param box1: (4, ), format (left, upper, right, bottom)
    :return: float, iou score
    """
    l_max = max(box1[0], box2[0])
    r_min = min(box1[2], box2[2])
    u_max = max(box1[1], box2[1])
    b_min = min(box1[3], box2[3])

    if l_max>=r_min or u_max>=b_min:
        return 0
    else:
        area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
        area_i = (b_min-u_max)*(r_min-l_max)
        return area_i/(area1+area2-area_i)

def xywh_iou(box1,box2):
    """
    calculate iou between box1 and box2
    :param box1: (4, ), format (left, upper, width, height)
    :param box2: (4, ), format (left, upper, width, height)
    :return: float, iou score
    """
    l_max = max(box1[0], box2[0])
    r_min = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
    u_max = max(box1[1], box2[1])
    b_min = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

    if l_max>=r_min or u_max>=b_min:
        return 0.
    else:
        area1 = box1[2]*box1[3]
        area2 = box2[2]*box2[3]
        area_i = (b_min-u_max)*(r_min-l_max)
        return area_i*1.0/(area1+area2-area_i)

def xyxy_to_xywh(xyxy):
    """
    convert box [left upper right bottom] to box [left upper width height].
    """
    if isinstance(xyxy, (list, tuple)):
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        if len(xyxy.shape) == 1:
            x1, y1 = xyxy[0], xyxy[1]
            w = xyxy[2] - x1 + 1
            h = xyxy[3] - y1 + 1
            return (x1, y1, w, h)
        elif len(xyxy.shape) == 2:
            return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
        else:
            raise
    else:
        raise TypeError

def xywh_to_xyxy(xywh):
    """
    convert box [left upper width height] to box [left upper right bottom].
    """
    if isinstance(xywh, (list, tuple)):
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = xywh[2] + x1
        y2 = xywh[3] + y1
        return (x1, y1, x2, y2)
    elif isinstance(xywh, np.ndarray):
        if len(xyxy.shape) == 1:
            assert len(xywh) == 4
            x1, y1 = xywh[0], xywh[1]
            x2 = xywh[2] + x1
            y2 = xywh[3] + y1
            return (x1, y1, x2, y2)
        elif len(xyxy.shape) == 2:
            return np.hstack((xywh[:, 0:2], xywh[:, 2:4] + xywh[:, 0:2]))
        else:
            raise
    else:
        raise TypeError

def msk_to_xywh(msk):
    """
    calculate box [left upper width height] from mask.
    :param msk: nd.array, single-channel or 3-channels mask
    :return: float, iou score    
    """
    if len(msk.shape) > 2:
        msk = msk[..., 0]
    nonzeros = np.nonzero(msk.astype(np.uint8))
    u, l = np.min(nonzeros, axis=1)
    b, r = np.max(nonzeros, axis=1)
    return np.array((l, u, r-l+1, b-u+1))

def msk_to_xyxy(msk):
    """
    calculate box [left upper right bottom] from mask.
    :param msk: nd.array, single-channel or 3-channels mask
    :return: float, iou score    
    """
    if len(msk.shape) > 2:
        msk = msk[..., 0]
    nonzeros = np.nonzero(msk.astype(np.uint8))
    u, l = np.min(nonzeros, axis=1)
    b, r = np.max(nonzeros, axis=1)
    return np.array((l, u, r+1, b+1))

def get_edges(msk):
    """
    get edge from mask
    :param msk: nd.array, single-channel or 3-channel mask
    :return: edges: nd.array, edges with same shape with mask
    """
    msk_sp = msk.shape
    if len(msk_sp) == 2:
        c = 1 # single channel
    elif (len(msk_sp) == 3) and (msk_sp[2] == 3):
        c = 3 # 3 channels
        msk = msk[:, :, 0] != 0        
    edges = np.zeros(msk_sp[:2])
    edges[:-1, :] = np.logical_and(msk[:-1, :] != 0, msk[1:, :] == 0) + edges[:-1, :]
    edges[1:, :] = np.logical_and(msk[1:, :] != 0, msk[:-1, :] == 0) + edges[1:, :]
    edges[:, :-1] = np.logical_and(msk[:, :-1] != 0, msk[:, 1:] == 0) + edges[:, :-1]
    edges[:, 1:] = np.logical_and(msk[:, 1:] != 0, msk[:, :-1] == 0) + edges[:, 1:]
    if c == 3:
        return np.dstack((edges, edges, edges))
    else:
        return edges

def zoom_in(im, c, s, res, channel=3, interpolate=cv2.INTER_LINEAR):
    """
    zoom in on the object with center c and size s, and resize to resolution res.
    :param im: nd.array, single-channel or 3-channel image
    :param c: (w, h), object center
    :param s: scalar, object size
    :param res: target resolution
    :param channel:
    :param interpolate:
    :return: zoomed object patch 
    """
    c_w, c_h = c
    c_w, c_h, s, res = int(c_w), int(c_h), int(s), int(res)
    if channel==1:
        im = im[..., None]
    h, w = im.shape[:2]
    u = int(c_h-0.5*s+0.5)
    l = int(c_w-0.5*s+0.5)
    b = u+s
    r = l+s
    if (u>=h) or (l>=w) or (b<=0) or (r<=0):
        return np.zeros((res, res, channel)).squeeze()
    if u < 0:
        local_u = -u
        u = 0 
    else:
        local_u = 0
    if l < 0:
        local_l = -l
        l = 0
    else:
        local_l = 0
    if b > h:
        local_b = s-(b-h)
        b = h
    else:
        local_b = s
    if r > w:
        local_r = s-(r-w)
    else:
        local_r = s
    im_crop = np.zeros((s, s, channel))
    im_crop[local_u:local_b, local_l:local_r, :] = im[u:b, l:r, :]
    im_crop = im_crop.squeeze()
    im_resize = cv2.resize(im_crop, (res, res), interpolation=interpolate)
    c_h = 0.5*(u+b)
    c_w = 0.5*(l+r)
    s = s
    return im_resize, c_h, c_w, s

def Crop_by_Pad(img, center, scale, res, channel=3, interpolation=cv2.INTER_NEAREST, resize=True):
    ht, wd = img.shape[0], img.shape[1]
    upper = max(0, int(center[0] - scale / 2. + 0.5))
    left  = max(0, int(center[1] - scale / 2. + 0.5))
    bottom = min(ht, int(center[0] - scale / 2. + 0.5) + int(scale))
    right  = min(wd, int(center[1] - scale / 2. + 0.5) + int(scale))
    crop_ht = float(bottom - upper)
    crop_wd = float(right - left)
    if crop_ht > crop_wd:
        resize_ht = res
        resize_wd = int(res / crop_ht * crop_wd + 0.5)
    elif crop_ht < crop_wd:
        resize_wd = res
        resize_ht = int(res / crop_wd * crop_ht + 0.5)
    else:
        resize_wd = resize_ht = res

    if channel == 3 or channel == 1:
        tmpImg = img[upper:bottom, left:right, :]
        if not resize:
            outImg = np.zeros((int(scale), int(scale), channel))
            outImg[int(scale / 2.0 - (bottom-upper) / 2.0 + 0.5):(int(scale / 2.0 - (bottom - upper) / 2.0 + 0.5) + (bottom-upper)),
            int(scale / 2.0 - (right-left) / 2.0 + 0.5):(int(scale / 2.0 - (right-left) / 2.0 + 0.5) + (right-left)), :] = tmpImg
            return outImg
        try:
            resizeImg = cv2.resize(tmpImg, (resize_wd, resize_ht), interpolation=interpolation)
        except:
            # raise Exception
            return np.zeros((res, res, channel))

        if len(resizeImg.shape) < 3:
            resizeImg = np.expand_dims(resizeImg, axis=2) # for depth image, add the third dimension
        outImg = np.zeros((res, res, channel))
        outImg[int(res / 2.0 - resize_ht / 2.0 + 0.5):(int(res / 2.0 - resize_ht / 2.0 + 0.5) + resize_ht),
        int(res / 2.0 - resize_wd / 2.0 + 0.5):(int(res / 2.0 - resize_wd / 2.0 + 0.5) + resize_wd), :] = resizeImg
    else:
        raise NotImplementedError

    return outImg

if __name__ == '__main__':
    im = np.random.randn(480, 640)
    c = (440, 500)
    s = 150
    res = 256
    im_zoom = zoom_in(im, c, s, res, channel=1)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im)
    plt.scatter(c[1], c[0], c='r', marker="o")

    plt.subplot(1,2,2)
    plt.imshow(np.squeeze(im_zoom))
    plt.show()
