import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
# from utils.fourier import FDA_source_to_target_np
# from datasets.utils.normalize import normalize_image
# from datasets.utils.slaug import LocationScaleAugmentation
from utils.fourier import FDA_source_to_target_np
from datasets.utils.normalize import normalize_image
from datasets.utils.slaug import LocationScaleAugmentation
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform

def prostate_transform(patch_size=(384, 384)):
    tr_transforms = []
    # tr_transforms.append(RandomCropTransform(crop_size=256, margins=(0, 0, 0), data_key="image", label_key="mask"))
    tr_transforms.append(
        SpatialTransform(
            patch_size, patch_center_dist_from_border=[i // 2 for i in patch_size],
            do_elastic_deform=True, alpha=(0., 900.), sigma=(9., 13.),
            do_rotation=True,
            angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            angle_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            do_scale=True, scale=(0.85, 1.25),
            border_mode_data='constant', border_cval_data=0,
            order_data=3, border_mode_seg="constant", border_cval_seg=-1,
            order_seg=1,
            random_crop=True,
            p_el_per_sample=0.2, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False,
            data_key="data", label_key="mask")
    )
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="data"))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5,
                              p_per_sample=0.2, data_key="data"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="data"))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="data"))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="data"))
    tr_transforms.append(
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0,
                                       order_upsample=3, p_per_sample=0.25,
                                       ignore_axes=None, data_key="data"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                                        p_per_sample=0.15, data_key="data"))

    tr_transforms.append(MirrorTransform(axes=(0, 1), data_key="data", label_key="mask"))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def fourier_augmentation_reverse(data, fda_beta=0.1):
    this_fda_beta = round(0.05+np.random.random() * fda_beta, 2)
    lowf_batch = data[::-1]
    fda_data = FDA_source_to_target_np(data, lowf_batch, L=this_fda_beta)
    return fda_data

def sl_augmentation(image, mask):
    location_scale = LocationScaleAugmentation(vrange=(0., 255.), background_threshold=0.01)
    GLA = location_scale.Global_Location_Scale_Augmentation(image.copy())
    LLA = location_scale.Local_Location_Scale_Augmentation(image.copy(), mask.copy().astype(np.int32))
    return GLA, LLA

def get_train_transform(patch_size=(512, 512)):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))
    # tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
    #                                            p_per_channel=0.5, p_per_sample=0.15))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def get_SpatialTransform_transform(patch_size=(512, 512)):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def get_structure_destroyed_transform(patch_size=(512, 512)):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def collate_fn_tr(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    tr_transforms = get_train_transform(patch_size=image.shape[-2:])
    data_dict = tr_transforms(**data_dict)
    return data_dict

def collate_fn_ts(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    return data_dict

def collate_fn_tr_only_sd_trans(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    tr_transforms = get_structure_destroyed_transform(patch_size=image.shape[-2:])
    data_dict = tr_transforms(**data_dict)
    return data_dict

def collate_fn_tr_styleaug(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    tr_transforms = get_train_transform(patch_size=image.shape[-2:])
    data_dict = tr_transforms(**data_dict)

    data_spatial = {'data': image, 'seg': label, 'name': name}
    spatialTransform = get_SpatialTransform_transform(patch_size=image.shape[-2:])
    data_spatial = spatialTransform(**data_spatial)
    data_dict['spatial_data'] = data_spatial['data']
    data_dict['spatial_data'] = normalize_image(data_dict['spatial_data'])
    data_dict['spatial_seg'] = data_spatial['seg']

    data_dict['clean_data']=np.copy(data_dict['data'])
    data_dict['data'] = normalize_image(data_dict['data'])

    fda_data = fourier_augmentation_reverse(data_dict['data'])
    data_dict['fda_data'] = normalize_image(fda_data)

    GLA, LLA = sl_augmentation(data_dict['data'], data_dict['seg'])
    data_dict['GLA'] = normalize_image(GLA)
    data_dict['LLA'] = normalize_image(LLA)
    return data_dict

# def get_train_transform_w(patch_size=(384, 384)):
#     tr_transforms = []
#     # tr_transforms.append(RandomCropTransform(crop_size=256, margins=(0, 0, 0), data_key="image", label_key="mask"))
#     tr_transforms.append(
#         SpatialTransform(
#             patch_size, patch_center_dist_from_border=[i // 2 for i in patch_size],
#             do_elastic_deform=True, alpha=(0., 900.), sigma=(9., 13.),
#             do_rotation=True,
#             angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
#             angle_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
#             do_scale=True, scale=(0.85, 1.25),
#             border_mode_data='constant', border_cval_data=0,
#             order_data=3, border_mode_seg="constant", border_cval_seg=-1,
#             order_seg=1,
#             random_crop=True,
#             p_el_per_sample=0.2, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
#             independent_scale_for_each_axis=False,
#             data_key="data", label_key="mask")
#     )
#     tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="data"))
#     tr_transforms.append(
#         GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5,
#                               p_per_sample=0.2, data_key="data"))
#     tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="data"))
#     tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="data"))
#     tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="data"))
#     tr_transforms.append(
#         SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0,
#                                        order_upsample=3, p_per_sample=0.25,
#                                        ignore_axes=None, data_key="data"))
#     tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
#                                         p_per_sample=0.15, data_key="data"))
#
#     tr_transforms.append(MirrorTransform(axes=(0, 1), data_key="data", label_key="mask"))
#
#     # now we compose these transforms together
#     tr_transforms = Compose(tr_transforms)
#     return tr_transforms
#
# def collate_fn_w_transform(batch):
#     image, label, name = zip(*batch)
#     image = np.stack(image, 0)
#     label = np.stack(label, 0)
#     name = np.stack(name, 0)
#     data_dict = {'data': image, 'mask': label, 'name': name}
#     tr_transforms = get_train_transform_w()
#     data_dict = tr_transforms(**data_dict)
#     data_dict['data'] = np.repeat(data_dict['data'], 3, axis=1)
#     fda_data = fourier_augmentation_reverse(data_dict['data'])
#     data_dict['fda_data'] = normalize_image(fda_data)
#     GLA, LLA = sl_augmentation(data_dict['data'], data_dict['mask'])
#     data_dict['data'] = normalize_image(data_dict['data'])
#     data_dict['GLA'] = normalize_image(GLA)
#     data_dict['LLA'] = normalize_image(LLA)
#     return data_dict

def collate_prostate_styleaug(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'mask': label, 'name': name}
    tr_transforms = prostate_transform()
    data_dict = tr_transforms(**data_dict)
    data_dict['data'] = np.repeat(data_dict['data'], 3, axis=1)

    data_dict['clean_data'] = np.copy(data_dict['data'])
    data_dict['data'] = normalize_image(data_dict['data'])

    fda_data = fourier_augmentation_reverse(data_dict['data'])
    data_dict['fda_data'] = normalize_image(fda_data)
    GLA, LLA = sl_augmentation(data_dict['data'], data_dict['mask'])
    data_dict['GLA'] = normalize_image(GLA)
    data_dict['LLA'] = normalize_image(LLA)
    # data_dict['data'] = normalize_image(data_dict['data'])
    return data_dict

def collate_prostate_test(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'mask': label, 'name': name}
    data_dict['data'] = np.repeat(data_dict['data'], 3, axis=1)
    return data_dict


#
#
# import torchvision.transforms.functional as F
# import scipy.ndimage
# import random
# from PIL import Image
# import numpy as np
# import cv2
# import numbers
#
#
# class ToTensor(object):
#
#     def __call__(self, data):
#         image, label = data['image'], data['label']
#         return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}
#
# class Resize(object):
#
#     def __init__(self, size):
#         self.size = size
#
#     def __call__(self, data):
#         image, label = data['image'], data['label']
#
#         return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size)}
#
# class RandomHorizontalFlip(object):
#     def __init__(self, p=0.5):
#         self.p = p
#
#     def __call__(self, data):
#         image, label = data['image'], data['label']
#
#         if random.random() < self.p:
#             return {'image': F.hflip(image), 'label': F.hflip(label)}
#
#         return {'image': image, 'label': label}
#
# class RandomVerticalFlip(object):
#     def __init__(self, p=0.5):
#         self.p = p
#
#     def __call__(self, data):
#         image, label = data['image'], data['label']
#
#         if random.random() < self.p:
#             return {'image': F.vflip(image), 'label': F.vflip(label)}
#
#         return {'image': image, 'label': label}
#
# class RandomRotation(object):
#
#     def __init__(self, degrees, resample=False, expand=False, center=None):
#         if isinstance(degrees, numbers.Number):
#             if degrees < 0:
#                 raise ValueError("If degrees is a single number, it must be positive.")
#             self.degrees = (-degrees, degrees)
#         else:
#             if len(degrees) != 2:
#                 raise ValueError("If degrees is a sequence, it must be of len 2.")
#             self.degrees = degrees
#         self.resample = resample
#         self.expand = expand
#         self.center = center
#
#     @staticmethod
#     def get_params(degrees):
#         """Get parameters for ``rotate`` for a random rotation.
#
#         Returns:
#             sequence: params to be passed to ``rotate`` for random rotation.
#         """
#         angle = random.uniform(degrees[0], degrees[1])
#
#         return angle
#
#     def __call__(self, data):
#
#         """
#             img (PIL Image): Image to be rotated.
#
#         Returns:
#             PIL Image: Rotated image.
#         """
#         image, label = data['image'], data['label']
#
#         if random.random() < 0.5:
#             angle = self.get_params(self.degrees)
#             return {'image': F.rotate(image, angle, self.resample, self.expand, self.center),
#                     'label': F.rotate(label, angle, self.resample, self.expand, self.center)}
#
#         return {'image': image, 'label': label}
#
# class RandomZoom(object):
#     def __init__(self, zoom=(0.8, 1.2)):
#         self.min, self.max = zoom[0], zoom[1]
#
#     def __call__(self, data):
#         image, label = data['image'], data['label']
#
#         if random.random() < 0.5:
#             image = np.array(image)
#             label = np.array(label)
#
#             zoom = random.uniform(self.min, self.max)
#             zoom_image = clipped_zoom(image, zoom)
#             zoom_label = clipped_zoom(label, zoom)
#
#             zoom_image = Image.fromarray(zoom_image.astype('uint8'), 'RGB')
#             zoom_label = Image.fromarray(zoom_label.astype('uint8'), 'L')
#             return {'image': zoom_image, 'label': zoom_label}
#
#         return {'image': image, 'label': label}
#
# def clipped_zoom(img, zoom_factor, **kwargs):
#     h, w = img.shape[:2]
#
#     # For multichannel images we don't want to apply the zoom factor to the RGB
#     # dimension, so instead we create a tuple of zoom factors, one per array
#     # dimension, with 1's for any trailing dimensions after the width and height.
#     zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
#
#     # Zooming out
#     if zoom_factor < 1:
#
#         # Bounding box of the zoomed-out image within the output array
#         zh = int(np.round(h * zoom_factor))
#         zw = int(np.round(w * zoom_factor))
#         top = (h - zh) // 2
#         left = (w - zw) // 2
#
#         # Zero-padding
#         out = np.zeros_like(img)
#         out[top:top + zh, left:left + zw] = scipy.ndimage.zoom(img, zoom_tuple, **kwargs)
#
#     # Zooming in
#     elif zoom_factor > 1:
#
#         # Bounding box of the zoomed-in region within the input array
#         zh = int(np.round(h / zoom_factor))
#         zw = int(np.round(w / zoom_factor))
#         top = (h - zh) // 2
#         left = (w - zw) // 2
#
#         zoom_in = scipy.ndimage.zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)
#
#         # `zoom_in` might still be slightly different with `img` due to rounding, so
#         # trim off any extra pixels at the edges or zero-padding
#
#         if zoom_in.shape[0] >= h:
#             zoom_top = (zoom_in.shape[0] - h) // 2
#             sh = h
#             out_top = 0
#             oh = h
#         else:
#             zoom_top = 0
#             sh = zoom_in.shape[0]
#             out_top = (h - zoom_in.shape[0]) // 2
#             oh = zoom_in.shape[0]
#         if zoom_in.shape[1] >= w:
#             zoom_left = (zoom_in.shape[1] - w) // 2
#             sw = w
#             out_left = 0
#             ow = w
#         else:
#             zoom_left = 0
#             sw = zoom_in.shape[1]
#             out_left = (w - zoom_in.shape[1]) // 2
#             ow = zoom_in.shape[1]
#
#         out = np.zeros_like(img)
#         out[out_top:out_top + oh, out_left:out_left + ow] = zoom_in[zoom_top:zoom_top + sh, zoom_left:zoom_left + sw]
#
#     # If zoom_factor == 1, just return the input array
#     else:
#         out = img
#     return out
#
# # class Translation(object):
# #     def __init__(self, translation):
# #         self.translation = translation
#
# #     def __call__(self, data):
# #         image, label = data['image'], data['label']
#
# #         if random.random() < 0.5:
# #             image = np.array(image)
# #             label = np.array(label)
# #             rows, cols, ch = image.shape
#
# #             translation = random.uniform(0, self.translation)
# #             tr_x = translation / 2
# #             tr_y = translation / 2
# #             Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
#
# #             translate_image = cv2.warpAffine(image, Trans_M, (cols, rows))
# #             translate_label = cv2.warpAffine(label, Trans_M, (cols, rows))
#
# #             translate_image = Image.fromarray(translate_image.astype('uint8'), 'RGB')
# #             translate_label = Image.fromarray(translate_label.astype('uint8'), 'L')
#
# #             return {'image': translate_image, 'label': translate_label}
#
# #         return {'image': image, 'label': label}
#
# class RandomCrop(object):
#     def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#         self.padding = padding
#         self.pad_if_needed = pad_if_needed
#         self.fill = fill
#         self.padding_mode = padding_mode
#
#     @staticmethod
#     def get_params(img, output_size):
#         """Get parameters for ``crop`` for a random crop.
#         Args:
#             img (PIL Image): Image to be cropped.
#             output_size (tuple): Expected output size of the crop.
#         Returns:
#             tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
#         """
#         w, h = img.size
#         th, tw = output_size
#         if w == tw and h == th:
#             return 0, 0, h, w
#
#         # i = torch.randint(0, h - th + 1, size=(1, )).item()
#         # j = torch.randint(0, w - tw + 1, size=(1, )).item()
#         i = random.randint(0, h - th)
#         j = random.randint(0, w - tw)
#         return i, j, th, tw
#
#     def __call__(self, data):
#         """
#         Args:
#             img (PIL Image): Image to be cropped.
#         Returns:
#             PIL Image: Cropped image.
#         """
#         img, label = data['image'], data['label']
#         if self.padding is not None:
#             img = F.pad(img, self.padding, self.fill, self.padding_mode)
#             label = F.pad(label, self.padding, self.fill, self.padding_mode)
#         # pad the width if needed
#         if self.pad_if_needed and img.size[0] < self.size[1]:
#             img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
#             label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
#
#         # pad the height if needed
#         if self.pad_if_needed and img.size[1] < self.size[0]:
#             img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
#             label = F.pad(label, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
#         i, j, h, w = self.get_params(img, self.size)
#         img = F.crop(img, i, j, h, w)
#         label = F.crop(label, i, j, h, w)
#         return {"image": img, "label": label}
#
# class Normalization(object):
#
#     def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
#         self.mean = mean
#         self.std = std
#
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         image = F.normalize(image, self.mean, self.std)
#         return {'image': image, 'label': label}