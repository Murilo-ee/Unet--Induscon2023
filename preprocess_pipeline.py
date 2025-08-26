import torch
import torch.nn as nn
import cv2
import numpy as np  
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from patchify import patchify, unpatchify


class RemoveSpecularHighlights(nn.Module):
  def __init__(self,
               dilate_kernel: int = 5,
               inpaint_kernel: int = 10,
               mask_sensitivity: float = 0.65,
               plot_masks: bool = False,
               plot_results: bool = False):
    
    super().__init__()
    self.dilate_kernel = dilate_kernel
    self.inpaint_kernel = inpaint_kernel
    self.mask_sensitivity = mask_sensitivity
    self.plot_masks = plot_masks
    self.plot_results = plot_results
    self.plot_masks = plot_masks
    self.toPIL = transforms.ToPILImage()


  def dilate_mask(self,
                  mask: np.ndarray) -> np.ndarray: 

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.inpaint_kernel, self.inpaint_kernel))

    dilated_mask = cv2.dilate(mask.astype('uint8'), kernel, 
                      borderType=cv2.BORDER_CONSTANT, borderValue=int(0))

    if self.plot_masks:
      fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,6))

      ax1.imshow(mask, cmap='gray')
      ax2.imshow(dilated_mask, cmap='gray')

      ax1.title.set_text('Máscara')  # type:ignore
      ax2.title.set_text('Máscara Dilatada')  # type:ignore

    return dilated_mask


  def inpaint_img(self,
                  img: np.ndarray,
                  dilated_mask: np.ndarray) -> np.ndarray:
    
    inpainted_img = cv2.inpaint(img,dilated_mask,self.inpaint_kernel,cv2.INPAINT_TELEA)

    if self.plot_results:
      fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,6))

      ax1.imshow(img)
      ax2.imshow(inpainted_img)
 
      ax1.title.set_text('Original')  # type:ignore
      ax2.title.set_text('Corrigida')  # type:ignore
    
    return inpainted_img 


  def forward(self, img: torch.Tensor) -> np.ndarray:
    gray_img = transforms.functional.rgb_to_grayscale(img)  # type:ignore
    threshold = torch.mul(img.median(), 0.3)

    gray_img = torch.sub(gray_img,threshold)
    mask = torch.gt(gray_img,self.mask_sensitivity).to(torch.int32)
    mask_arr = mask.squeeze().numpy().astype('uint8')

    dilated_mask = self.dilate_mask(mask_arr)

    img_arr = np.array(self.toPIL(img))
    inpainted_img = self.inpaint_img(img_arr,dilated_mask)

    return inpainted_img
  


class LightingNormalization(nn.Module):
  """
  Lida com imagens no formato (H, W, C).
  """
  def __init__(self,
               weight: float = 0.3,
               patch_size: int = 16,
               plot_results: bool = False):
    
    super().__init__()
    self.weight = weight
    self.patch_size = patch_size
    self.plot_results = plot_results


  def plot(self,img,reconstructed_img):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,10))

    ax1.imshow(img)
    ax2.imshow(reconstructed_img)

    ax1.title.set_text('Original')  # type:ignore
    ax2.title.set_text(f'Normalizada (peso = {self.weight})')  # type:ignore


  def forward(self,img_arr: np.ndarray) -> np.ndarray:
    img_shape = img_arr.shape
    patches = patchify(img_arr, 
                       (self.patch_size, self.patch_size, 3),
                       step=self.patch_size)

    patch_i, patch_j, _, _, _, patch_c = patches.shape

    norm_patches = patches.copy()

    for c in range(patch_c):
      for i in range(patch_i):
        for j in range(patch_j):
          sub_value = np.median(patches[i,j,:,:,:,c]) * self.weight
          norm_patches[i,j,:,:,:,c] = patches[i,j,:,:,:,c] - sub_value

    reconstructed_img = unpatchify(norm_patches, img_shape)

    if self.plot_results:
      self.plot(img_arr,reconstructed_img)

    return reconstructed_img
  