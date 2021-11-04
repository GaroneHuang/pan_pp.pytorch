import os
import json
import numpy as np
import cv2
from PIL import Image
from numpy.random.mtrand import rand
import torch
from torch.utils import data
import torchvision.transforms as transforms
import Polygon as plg
import random
import pyclipper
from functools import reduce
from dataset.pan_pp.charset import charset


benchmark_root_path = "../datasets"
benchmark_pretrain_gt_dir = os.path.join(benchmark_root_path, "annotations_separate/full_pretrain")
benchmark_train_gt_dir = os.path.join(benchmark_root_path, "annotations_separate/full_train")
benchmark_val_gt_dir = os.path.join(benchmark_root_path, "annotations_separate/full_val")
benchmark_test_gt_dir = os.path.join(benchmark_root_path, "annotations_separate/full_test")

down_sample_rate = 4


def get_img(img_path, read_type='pil'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            if img is None:
                print('Cannot read image: %s.' % img_path)
                raise
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception:
        print('Cannot read image: %s.' % img_path)
        raise
    return img


def get_location(box, is_accept_poly):
    poly = box['poly']
    quad = box['quad']
    xywh_rect = box['xywh_rect']
    if not is_accept_poly:
        poly = []
    if len(poly) > 0 and len(poly)%2 == 0:
        loc = reduce(lambda x, y: x + y, poly)
    elif len(quad) == 8:
        loc = quad
    elif len(xywh_rect) == 4:
        x, y, w, h = xywh_rect
        loc = [x, y, x+w, y, x+w, y+h, x, y+h]
    else:
        loc = None
    return loc

def get_ann(img, gt):
    h, w = img.shape[0:2]
    bboxes = []
    words = []
    for granularity in gt['annotations']:
        for box in gt['annotations'][granularity]:
            if box['anno_cat'] != 'standard':
                continue
            loc = get_location(box, True)
            if loc is None:
                continue
            word = box['transcript']
            if box['ignore'] == 1:
                words.append('###')
            else:
                words.append(word)
            # bbox = np.array(loc) / ([w * 1.0, h * 1.0] * 4)
            bbox = np.array(loc, dtype=np.float)
            bbox[::2] /= w * 1.0
            bbox[1::2] /= h * 1.0
            bboxes.append(bbox)
    return bboxes, words


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img,
                                      rotation_matrix, (h, w),
                                      flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs


def random_scale(img, min_sizes, max_sizes):
    min_size = random.choice(min_sizes)
    max_size = random.choice(max_sizes)
    h, w = img.shape[:2]
    scale = min_size / min(w, h)
    if h < w:
        neww, newh = scale * w, min_size
    else:
        neww, newh = min_size, scale * h
    if max(neww, newh) > max_size:
        scale = max_size / max(neww, newh)
        neww = neww * scale
        newh = newh * scale
    neww = int(round(neww / 32.) * 32.)
    newh = int(round(neww / 32.) * 32.)
    img = cv2.resize(img, dsize=(neww, newh))
    return img


def random_crop_padding(imgs, target_size):
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img,
                                       0,
                                       p_h - t_h,
                                       0,
                                       p_w - t_w,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0
                                                   for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img,
                                       0,
                                       p_h - t_h,
                                       0,
                                       p_w - t_w,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=(0, ))
        n_imgs.append(img_p)
    return n_imgs


def update_word_mask(instance, instance_before_crop, word_mask):
    labels = np.unique(instance)

    for label in labels:
        if label == 0:
            continue
        ind = instance == label
        if np.sum(ind) == 0:
            word_mask[label] = 0
            continue
        ind_before_crop = instance_before_crop == label
        # print(np.sum(ind), np.sum(ind_before_crop))
        if float(np.sum(ind)) / np.sum(ind_before_crop) > 0.9:
            continue
        word_mask[label] = 0

    return word_mask


def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5),
                         max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox)[0]
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception:
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes


def get_vocabulary(EOS='EOS', PADDING='PAD', UNKNOWN='UNK'):
    voc = list(charset)

    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char

class PAN_PP_BENCHMARK(data.Dataset):
    def __init__(self,
                 split=("train", ),
                 is_train=True,
                 is_transform=False,
                 img_size=None,
                 min_sizes=(640, 672, 704),
                 max_sizes=(1600, ),
                 kernel_scale=0.5,
                 with_rec=False,
                 read_type='pil',
                 report_speed=False):
        self.split = split
        self.is_train = is_train
        self.is_transform = is_transform

        self.img_size = img_size if (
            img_size is None or isinstance(img_size, tuple)) else (img_size,
                                                                   img_size)
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.kernel_scale = kernel_scale
        self.with_rec = with_rec
        self.read_type = read_type

        gt_dirs = []
        if "pretrain" in split:
            gt_dirs.append(benchmark_pretrain_gt_dir)
        if "train" in split:
            gt_dirs.append(benchmark_train_gt_dir)
        if "val" in split:
            gt_dirs.append(benchmark_val_gt_dir)
        if "test" in split:
            gt_dirs.append(benchmark_test_gt_dir)
        if len(gt_dirs) <= 0:
            print('Error: split must be pretrain, train, val or test!')
            raise

        self.gt_paths = []
        for gt_dir in gt_dirs:
            gt_names = os.listdir(gt_dir)
            self.gt_paths += list(map(lambda x: os.path.join(gt_dir, x), gt_names))

        if report_speed:
            target_size = 3000
            extend_scale = (target_size + len(self.gt_paths) - 1) // len(
                self.gt_paths)
            self.gt_paths = (self.gt_paths * extend_scale)[:target_size]

        self.voc, self.char2id, self.id2char = get_vocabulary()
        self.max_word_num = 200
        self.max_word_len = 32
        print('reading type: %s.' % self.read_type)

    def __len__(self):
        return len(self.gt_paths)

    def prepare_train_data(self, index):
        gt_path = self.gt_paths[index]

        f_gt = open(gt_path, "r")
        gt = json.load(f_gt)
        f_gt.close()

        img = get_img(os.path.join(benchmark_root_path, gt['name']))
        bboxes, words = get_ann(img, gt)

        if len(bboxes) > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]
            words = words[:self.max_word_num]

        gt_words = np.full((self.max_word_num + 1, self.max_word_len),
                           self.char2id['PAD'],
                           dtype=np.int32)
        word_mask = np.zeros((self.max_word_num + 1, ), dtype=np.int32)
        for i, word in enumerate(words):
            if word == '###':
                continue
            gt_word = np.full((self.max_word_len, ),
                              self.char2id['PAD'],
                              dtype=np.int)
            for j, char in enumerate(word):
                if j > self.max_word_len - 1:
                    break
                if char in self.char2id:
                    gt_word[j] = self.char2id[char]
                else:
                    gt_word[j] = self.char2id['UNK']
            if len(word) > self.max_word_len - 1:
                gt_word[-1] = self.char2id['EOS']
            else:
                gt_word[len(word)] = self.char2id['EOS']
            gt_words[i + 1] = gt_word
            word_mask[i + 1] = 1

        if self.is_transform:
            img = random_scale(img, self.min_sizes, self.max_sizes)

        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if len(bboxes) > 0:
            for i in range(len(bboxes)):
                bboxes[i][::2] *= img.shape[1]
                bboxes[i][1::2] *= img.shape[0]
                bboxes[i] = bboxes[i].astype(np.int32).reshape(-1, 2)
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        gt_kernels = []
        for rate in [self.kernel_scale]:
            gt_kernel = np.zeros(img.shape[0:2], dtype=np.uint8)
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        if self.is_transform:
            imgs = [img, gt_instance, training_mask]
            imgs.extend(gt_kernels)

            imgs = random_rotate(imgs)
            gt_instance_before_crop = imgs[1].copy()
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernels = imgs[0], imgs[
                1], imgs[2], imgs[3:]
            word_mask = update_word_mask(gt_instance, gt_instance_before_crop,
                                         word_mask)

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        max_instance = np.max(gt_instance)
        gt_bboxes = np.zeros((self.max_word_num + 1, 4), dtype=np.int32)
        for i in range(1, max_instance + 1):
            ind = gt_instance == i
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))
            tl = np.min(points, axis=0)
            br = np.max(points, axis=0) + 1
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])

        img = Image.fromarray(img)
        img = img.convert('RGB')
        if self.is_transform:
            img = transforms.ColorJitter(brightness=32.0 / 255,
                                         saturation=0.5)(img)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)
        gt_text = torch.from_numpy(gt_text).long()
        gt_kernels = torch.from_numpy(gt_kernels).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        gt_bboxes = torch.from_numpy(gt_bboxes).long()
        gt_words = torch.from_numpy(gt_words).long()
        word_mask = torch.from_numpy(word_mask).long()

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
            gt_instances=gt_instance,
            gt_bboxes=gt_bboxes,
        )
        if self.with_rec:
            data.update(dict(gt_words=gt_words, word_masks=word_mask))

        return data

    def prepare_test_data(self, index):
        gt_path = self.gt_paths[index]

        f_gt = open(gt_path, "r")
        gt = json.load(f_gt)
        f_gt.close()

        img = get_img(os.path.join(benchmark_root_path, gt['name']))
        img_meta = dict(org_img_size=np.array(img.shape[:2]))

        img = random_scale(img, self.min_sizes, self.max_sizes)
        img_meta.update(dict(img_size=np.array(img.shape[:2])))

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)

        data = dict(imgs=img, img_metas=img_meta)

        return data

    def __getitem__(self, index):
        if self.is_train:
            return self.prepare_train_data(index)
        else:
            return self.prepare_test_data(index)
