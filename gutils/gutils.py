"""
This file provides all kinds of useful and general util functions or class. 
other utils which are dependent of your projects should not be included in this file.
"""


import cv2
import sys
import os
import base64
import numpy as np
import os.path as osp
import math
import json
import shutil
import random
import time
import glob
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, ALL_COMPLETED, as_completed
from pprint import pprint
from PIL import Image
from  matplotlib import pyplot as plt
from PIL.ExifTags import TAGS



class TimeEvaluator(object):
    def __init__(self, with_cuda=False):
        self.beg_time = None
        self.with_cuda = with_cuda

    def reset(self):
        if self.with_cuda:
            torch.cuda.synchronize()
        self.beg_time = time.perf_counter()

    def elapsed_time(self):
        if self.beg_time is None:
            return 0
        if self.with_cuda:
            torch.cuda.synchronize()
        req_end = time.perf_counter()
        elapse = round(((req_end - self.beg_time) * 1000), 3)
        self.beg_time = None
        return elapse


# 找到文件夹下所有指定后缀名的文件， 可选择要绝对路径还是相对路径
def find_imgs(dst_dir, ext=('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'), abs_pth=True):
    ret = []
    for root, dirs, files in os.walk(dst_dir):
        if not abs_pth:
            root = root[len(dst_dir)+1:]
        # print([osp.splitext(f)[-1] for f in files] )
        ret.extend([osp.join(root, f) for f in files if osp.splitext(f)[-1] in ext])
    return ret


def pth_exclude(ppth, cpth):
    """
    get rid of ppth from cpth
    :param ppth:
    :param cpth:
    :return:
    """
    if  ppth.endswith('/'):
        ppth = ppth[:-1]
    # else:
    #     _ppth = ppth
    idx = cpth.find(ppth)
    if idx == -1:
        return cpth
    else:
        return cpth[idx+len(ppth)+1:]



def b64str2ndarray(b64_str):
    img_data = base64.b64decode(b64_str)
    n_parr = np.frombuffer(img_data, np.uint8)
    img_np = cv2.imdecode(n_parr, cv2.IMREAD_COLOR)
    return img_np

# 注意， 使用 cv2.imencode 进行jpg压缩，即使将参数设置为100， 也会使得输出的二进制数据（即函数内的buffer）与原二进制数据有差异
def ndarray2b64str(img_np):
    retval, buffer = cv2.imencode('.jpg', img_np,  [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    pic_str = base64.b64encode(buffer) #bytes data
    pic_str = pic_str.decode()  # str
    return pic_str


def imgfile2b64str(img_path):
    with open(img_path, "rb") as image_file:
        raw_data = image_file.read()
        # print('read data: ', raw_data)
        encoded_image = base64.b64encode(raw_data) # bytes data
        # print('encode bytes data: ', encoded_image)
        req_file = encoded_image.decode('utf-8')  # str
        # print('encode str data: ', req_file)
    return req_file


def b64str2imgfile(b64_str, img_path):
    with open(img_path, 'wb') as f:
        b64data = b64_str.encode()
        img_data = base64.b64decode(b64data)
        f.write(img_data)



def base64_to_image(data, flag=1):
    if len(data) <= 0:
        return None
    return cv2.imdecode(np.fromstring(base64.b64decode(data + '=' * (-len(data) % 4)), np.uint8), flag)


def binary_to_base64(data):
    return base64.b64encode(data).decode('ascii')


def base64_to_binary(data):
    return base64.b64decode(data + '=' * (-len(data) % 4))


def image_to_base64(image, ext='.jpg'):
    _, data = cv2.imencode(ext, image)
    return binary_to_base64(data)

def binary_to_image(data, flag=1):
    arr = np.asarray(bytearray(data), dtype=np.uint8)
    return cv2.imdecode(arr, flag)


def image_to_binary(image, ext='.jpg'):
    _, data = cv2.imencode(ext, image)
    # same as data.tobytes()
    return data.tostring()


def numpy_to_string(data):
    f = io.BytesIO()
    np.save(f, data)
    # level 1 is the fastest and yields the lowest level of compression.
    # Level 9 is the slowest, yet it yields the highest level of compression.
    return binary_to_base64(f.getvalue())


def numpy_from_string(data):
    return np.load(io.BytesIO(base64_to_binary(data)))



def getexif_data_and_rotate_back(img_f, save_back=False):
    """
    获取图像的exif 信息（若有）， 删除exif信息， 并根据exif orientation信息转正
    :param img_f:
    :param save_back: True 的话按照img_f 路径存储转正后的结果
    :return: 转正后的图 ， PILimage对象. 如果img_f 无效或不存在 exif 信息， 返回None
    """
    from PIL.ExifTags import TAGS
    from PIL import Image, ImageOps
    # exif = dict()
    img = Image.open(img_f)
    if img is  None or img._getexif() is None:
        return
    exif = {TAGS[k]: v for k, v in img._getexif().items() if k in TAGS}
    orien = exif.get('Orientation', None)
    if orien is not None:
        print(f'get orien {orien}')
        if orien == 0: # ？ 
            img_ret = img
        if orien == 1:
            img_ret = img
        if orien == 2:
            img_ret = ImageOps.mirror(img)
        if orien == 3:
            img_ret = img.rotate(180, expand=True)
        if orien == 4:
            img_ret = img.rotate(180, expand=True)
            img_ret = ImageOps.mirror(img_ret)
        if orien == 5:
            img_ret = img.rotate(270, expand=True)
            img_ret = ImageOps.mirror(img_ret)
        if orien == 6:
            img_ret = img.rotate(270, expand=True)
        if orien == 7:
            img_ret = img.rotate(90, expand=True)
            img_ret = ImageOps.mirror(img_ret)
        if orien == 8:
            img_ret = img.rotate(90, expand=True)
    else:
        img_ret = img

    if save_back:
        img_ret.save(img_f)
    return img_ret




# print_var(a=a, b=b)
def print_var(**var_name):
    d = dict()
    d.update(**var_name)
    for k, v in d.items():
        print(f'{k}={v},\t')
    print()


def pil_to_cv(pil_image):
    """
    转换PIL的图像为OpenCV的图像
    :param pil_image: PIL Image
    :return: cv2 Mat
    """
    cv_image = np.array(pil_image)
    if len(cv_image.shape) == 2:
        # gray
        pass
    elif cv_image.shape[-1] == 3:
        # Convert RGB to BGR
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    elif cv_image.shape[-1] == 4:
        # Convert RGBA to BGRA
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGRA)
    else:
        raise Exception('unsupported image format')
    return cv_image


def cv_to_pil(cv_image):
    """
    转换OpenCV的图像为PIL的图像
    :param cv_image: cv2 Mat
    :return: PIL Image
    """
    cv_image = cv_image.astype('uint8')
    if len(cv_image.shape) == 2:
        # gray
        pil_image = Image.fromarray(cv_image)
    elif cv_image.shape[-1] == 3:
        # Convert BGR to RGB
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    elif cv_image.shape[-1] == 4:
        # Convert BGRA to RGBA
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA))
    else:
        raise Exception('unsupported image format')
    return pil_image


def make_dir(dirp, remove_if_exist=False):
    if osp.exists(dirp):
        if remove_if_exist:
            shutil.rmtree(dirp)
    os.makedirs(dirp, exist_ok=True)


def draw_box_info(im, bbx, color, text=None, xyxy=True):
    if xyxy:
        x, y, x1, y1 = list(map(round, bbx))
    else:
        x, y, w, h = list(map(round, bbx))
        x1, y1 = x+w, y+h
    thickness = 1
    # print(text)
    cv2.rectangle(im, (x, y), (x1, y1), color, thickness=thickness)
    if text is not None:
            cv2.putText(im, text, (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
    return im


def mask_color(src, mask, color, alpha=0.5):
    """
    在src图中指定位置指定颜色着色，形成透明颜色层的效果
    :param src: BGR原图
    :param mask: 单通道， 表示要做透明着色的地方
    :param color:
    :param alpha: 透明度， 越高颜色越深
    :return:
    """
    ret = src.copy()
    ret[mask>0] = color
    ret = cv2.addWeighted(src, 1-alpha, ret, alpha, 0)
    return ret


def multi_mask_color(src, mask, color_tab, alpha=0.5):
    """
    在src图中指定位置指定颜色着色，形成透明颜色层的效果
    :param src: BGR原图
    :param mask: 单通道， 表示要做透明着色的地方
    :param color_tab:
    :param alpha: 透明度， 越高颜色越深
    :return:
    """
    ret = src.copy()
    for val, color in color_tab.items():
        if 0<=val<=255:
            ret[mask==val] = color
    ret = cv2.addWeighted(src, 1-alpha, ret, alpha, 0)
    return ret


def tile_ims(ims, color=True):
    h, w = ims[0].shape[:2]
    if color:
        ret = np.zeros((h, w*len(ims), 3), dtype=ims[0].dtype)
    else:
        ret = np.zeros((h, w*len(ims)), dtype=ims[0].dtype)

    for i, im in enumerate(ims):
        ret[:, i*w:(i+1)*w] = im
    return ret


def tile_ims1(ims, color=True, style=0):
    assert len(ims) !=0
    hs, ws = [im.shape[0] for im in ims], [im.shape[1] for im in ims]
  
    if style==0:
        H = np.array(hs).max()
        W = np.array(ws).sum()
        ret = np.zeros((H, W), dtype=ims[0].dtype)
    elif style == 1:
        H = np.array(hs).sum()
        W = np.array(ws).max()
        ret = np.zeros((H, W), dtype=ims[0].dtype)
    if color:
        ret =np.expand_dims(ret, 2).repeat(3, axis=2)
    
    offset=0
    for i, im in enumerate(ims):
        h,w = im.shape[:2]
        if style == 0:
            ret[:h, offset:offset+w] = im
            offset+=w
        elif style == 1:
            ret[offset:offset+h, :w]= im
            offset += h        
    return ret




def get_timestamp():
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    return timestamp


class Logger(object):
    """
    Source https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    兼容with语义的logger  支持记录日志信息到本地文件
    
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.fpath = fpath
        if fpath is not None:
            make_dir(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg+'\n')
        if self.file is not None:
            self.file.write(msg+'\n')

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def plt_fig2np(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    # 重构成w h 4(argb)图像
    buf.shape = (w, h, 4)
    # 转换为 RGBA
    buf = np.roll(buf, 3, axis=2)
    # 得到 Image RGBA图像对象
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    # 转换为numpy array rgba四通道数组
    image = np.asarray(image)
    # 转换为rgb图像
    rgb_image = image[:, :, :3]
    return rgb_image



def check_tensor(**kwargs):
    for k, v in kwargs.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            print(f'{k} shape: {v.shape}')
        else:
            print(f'{k} is not a tensor')
    print()
    

def print_tensor(**kwargs):
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            print(f'{k}: {v}')
        else:
            print(f'{k} is not a tensor')
    print()
    
    
def check_var(**kwargs):
    for k, v in kwargs.items():
        print(f'{k}: {v}')
    print()





def x_rotate(anglex):
        arcx = math.pi * anglex/180
        return np.array([[1,0,0],
                         [0, math.cos(arcx), -math.sin(arcx)],
                         [0, math.sin(arcx), math.cos(arcx)]])
    
def y_rotate(angley):
    arcy = math.pi * angley/180
    return np.array([[math.cos(arcy), 0, math.sin(arcy)],
                        [0, 1, 0],
                        [-math.sin(arcy), 0, math.cos(arcy)]])
    
def z_rotate(anglez):
    arcz = math.pi * anglez/ 180
    return np.array([[math.cos(arcz), -math.sin(arcz), 0],
                        [math.sin(arcz), math.cos(arcz), 0],
                        [0,0,1]])


def calc_rotate_matrix_with_angels(angles, order='XYZ', internal_rotation=True):
    """
    根据旋转角计算旋转矩阵，angles 为3 elements list，分别表示旋转角度(非弧度)。
    每个位置表示哪个轴的角度，以及旋转顺序，由order指定。
    如order =XYZ 表示这三个角分别是XYZ轴的转角，顺序是X-Y-Z
    旋转角度的定义： 视线与某个轴延长的方向，逆时针为正， 顺时针为负
    internal_rotation 表示是否内旋，默认True
    概念参考：https://zhuanlan.zhihu.com/p/85108850
    """
    
    assert len(angles) == 3
    
    funcs = {index : func for func, index in zip([x_rotate, y_rotate,z_rotate], "XYZ")}
    if not internal_rotation:  # 如果外旋 ， order反序
        order = order[::-1]
    ret =  np.identity(3)
    check_var(order=order)
    for idx, axis in enumerate(order):
        func = funcs.get(axis)
        matrix = func(angles[idx])
        ret = ret @ matrix
        check_var(axis=axis, a=angles[idx])
    return ret    



def profile_latency(model, inps, cuda=True, times=20, warm_ratio=0.3):
    assert times>3, f'{times} must > 3'
    if cuda:
        assert torch.cuda.is_available()
        print(f'IN DEVICE {torch.cuda.get_device_name(0)}')
        inps  = list(inps)
        for i, inp in enumerate(inps):
            if isinstance(inp, (list, tuple)):
                inp = [i.cuda() for i in inp]
            else:
                inp = inp.cuda()
            inps[i] = inp
        model.cuda()
    tvr = TimeEvaluator(cuda)
    latencys = list()
    for i in range(times):
        tvr.reset()
        model(*inps)
        latency = tvr.elapsed_time()
        latencys.append(latency)
    warm_beg = int(times*warm_ratio)
    mean_latency = sum(latencys[warm_beg:])/ len(latencys[warm_beg:])
    print(f'run {times} times, mean_latency {mean_latency}ms. details:')
    pprint(latencys)      
    
    
def check_list_tensor(**kwargs):
    for name, l in kwargs.items():
        print(f'{name}: {len(l)} items')
        for i, item  in enumerate(l):
            if isinstance(item, (torch.Tensor, np.ndarray)):
                print(f'{i} shape: {item.shape}')
                
                
from multiprocessing import Pool
def tqdm_pool_worker(nworker, func, list_args, desc=''):
    with Pool(nworker) as p:
        for _ in tqdm(p.imap_unordered(func, list_args), total=len(list_args), desc=desc):
            pass
        
        
def add_indent_json(path):
    with open(path) as f:
        dat = json.load(f)
    with open(path, 'w') as f:
        json.dump(dat, f, indent=2, ensure_ascii=False)


def run_time(func):
    """
    获取运行耗时的装饰器
    Args:
        func (_type_): _description_
    """
    tvr = TimeEvaluator(False)
    def inner(*args, **kwargs):
        tvr.reset()
        result = func(*args, **kwargs)
        print(f'cost {tvr.elapsed_time()}ms')
        return result

    return inner


def plt_show_ims(pil_style_ims):
    """
    逐行显示多张图片
    Args:
        pil_style_ims (_type_): list of RGB image
    """
    N = len(pil_style_ims)
    plt.figure()
    for i in range(N):
        plt.subplot(N, 1, i+1)
        plt.imshow(pil_style_ims[i])
    plt.show()
