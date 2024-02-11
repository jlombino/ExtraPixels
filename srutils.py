import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset

def load_single_image(directory, image_number):
    
    files = sorted(os.listdir(directory))

    full_res_image = read_image(directory + files[image_number]).to(torch.int)
    full_res_image = (full_res_image - 128) / 128
    image_height = full_res_image.shape[1] 
    image_width = full_res_image.shape[2] 
    small_image = transforms.Resize((image_height//4, image_width//4), 
                                    antialias = False)(full_res_image)
    
    return small_image, full_res_image

class ImageDataset(Dataset):

    def __init__(self, directory, small_patch_size):

        super().__init__()
        self.directory = directory
        self.files = os.listdir(directory)
        self.small_patch_size = small_patch_size
        self.large_patch_size = small_patch_size * 4

    def __len__(self):
        
        return len(self.files)

    def __getitem__(self, index):

        full_res_image = (read_image(self.directory + '/' + self.files[index])).to(torch.int)
        full_res_image = (full_res_image - 128) / 128
        large_image = transforms.RandomCrop((self.large_patch_size,self.large_patch_size))(full_res_image)
        large_image = transforms.RandomVerticalFlip()(large_image)
        large_image = transforms.RandomHorizontalFlip()(large_image)
        small_image = transforms.Resize((self.small_patch_size, self.small_patch_size),
                                        antialias = False)(large_image)
        
        return small_image, large_image

def compare_model(generator, dataset_dir, device, index):

    fig, ax = plt.subplots(4, 4, figsize=(16,16))

    for row in ax:
        for image in row:
            image.set_xticks([])
            image.set_yticks([])
    
    small, large = load_single_image(dataset_dir, index)

    small_height = small.shape[1] 
    small_width = small.shape[2] 

    bilinear = transforms.Resize((small_height * 4, small_width * 4), 
                                antialias = True)(small)
    generated = generator(torch.unsqueeze(small,0).to(device)).to('cpu')
    generated = torch.squeeze(generated)
    generated = generated.detach()

    crop_locations = [400, 800, 1200]
    crop_small_size = 32
    crop_large_size = 128

    small_crops = [
        transforms.functional.crop(small,
                                  top = loc//4,
                                  left = loc//4,
                                  height = crop_small_size,
                                  width = crop_small_size)
        for loc in crop_locations]
    small_crops = [(torch.movedim(img,0,-1) * 128 + 128).to(torch.int)
                   for img in small_crops]
    
    large_crops = [
        transforms.functional.crop(large,
                                  top = loc,
                                  left = loc,
                                  height = crop_large_size,
                                  width = crop_large_size)
        for loc in crop_locations]
    large_crops = [(torch.movedim(img,0,-1) * 128 + 128).to(torch.int)
                   for img in large_crops]
        
    bilinear_crops = [
        transforms.functional.crop(bilinear,
                                  top = loc,
                                  left = loc,
                                  height = crop_large_size,
                                  width = crop_large_size)
        for loc in crop_locations]
    bilinear_crops = [(torch.movedim(img,0,-1) * 128 + 128).to(torch.int)
                   for img in bilinear_crops]
 
    generated_crops = [
        transforms.functional.crop(generated,
                                  top = loc,
                                  left = loc,
                                  height = crop_large_size,
                                  width = crop_large_size)
        for loc in crop_locations]
    generated_crops = [(torch.movedim(img,0,-1) * 128 + 128).to(torch.int)
                   for img in generated_crops]   
    
    small = (torch.movedim(small,0,-1) * 128 + 128).to(torch.int)
    large = (torch.movedim(large,0,-1) * 128 + 128).to(torch.int)
    bilinear = (torch.movedim(bilinear,0,-1) * 128 + 128).to(torch.int)
    generated = (torch.movedim(generated,0,-1) * 128 + 128).to(torch.int)
        
    ax[0][0].imshow(small)
    ax[0][0].set_title('Small')
    ax[0][1].imshow(bilinear)
    ax[0][1].set_title('Bilinear')
    ax[0][2].imshow(generated)
    ax[0][2].set_title('Neural Net')
    ax[0][3].imshow(large)
    ax[0][3].set_title('Original')

    for row, axis in enumerate(ax[1:]):
        axis[0].imshow(small_crops[row])
        axis[0].set_title('Small')
        axis[1].imshow(bilinear_crops[row])
        axis[1].set_title('Bilinear')
        axis[2].imshow(generated_crops[row])
        axis[2].set_title('Neural Net')
        axis[3].imshow(large_crops[row])
        axis[3].set_title('Original')
    
    return fig






