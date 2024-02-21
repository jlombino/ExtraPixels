import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.io import read_image, write_png
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

def crop_images(image, locations, size):
     
    crops = [image]
    for location in locations:
        crops.append(transforms.functional.crop(image,
                                      top = location,
                                      left = location,
                                      height = size,
                                      width = size))
    return crops

def load_image_generate(generator, dataset_dir, index):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    small, large = load_single_image(dataset_dir, index)

    large_height = large.shape[1] 
    large_width = large.shape[2] 

    bilinear = transforms.Resize((large_height, large_width), 
                                antialias = True)(small)
    generated = generator(torch.unsqueeze(small,0).to(device)).to('cpu')
    generated = torch.squeeze(generated)
    generated = generated.detach()
    
    small = transforms.Resize((large_height, large_width), 
                                antialias = False,
                                interpolation = transforms.InterpolationMode.NEAREST)(small)
    
    return small, bilinear, generated, large

def compare_model(generator, dataset_dir, index):

    fig, ax = plt.subplots(4, 4, figsize=(16,16))

    crop_locations = [400, 800, 1200]
    crop_size = 128

    small, bilinear, generated, large = load_image_generate(generator, dataset_dir, index)

    small_crops = crop_images(small, crop_locations, crop_size)
    small_crops = [(torch.movedim(image,0,-1) * 128 + 128).to(torch.int)
           for image in small_crops]
    
    bilinear_crops = crop_images(bilinear, crop_locations, crop_size)
    bilinear_crops = [(torch.movedim(image,0,-1) * 128 + 128).to(torch.int)
           for image in bilinear_crops]
    
    generated_crops = crop_images(generated, crop_locations, crop_size)
    generated_crops = [(torch.movedim(image,0,-1) * 128 + 128).to(torch.int)
           for image in generated_crops]
    
    large_crops = crop_images(large, crop_locations, crop_size)
    large_crops = [(torch.movedim(image,0,-1) * 128 + 128).to(torch.int)
           for image in large_crops]
    
        
    for row, axis in enumerate(ax):
        axis[0].imshow(small_crops[row])
        axis[0].set_title('Small')
        axis[1].imshow(bilinear_crops[row])
        axis[1].set_title('Bilinear')
        axis[2].imshow(generated_crops[row])
        axis[2].set_title('Neural Net')
        axis[3].imshow(large_crops[row])
        axis[3].set_title('Original')

    for row in ax:
        for axis in row:
            axis.set_xticks([])
            axis.set_yticks([])
    
    return fig

def save_val_images(generator, dataset_dir, index, label, save_all):

    crop_locations = [400, 1200]
    crop_size = 128

    small, bilinear, generated, large = load_image_generate(generator, dataset_dir, index)

    small_crops = crop_images(small, crop_locations, crop_size)
    small_crops = [(image * 128 + 128).to(torch.uint8)
           for image in small_crops]
    
    bilinear_crops = crop_images(bilinear, crop_locations, crop_size)
    bilinear_crops = [(image * 128 + 128).to(torch.uint8)
           for image in bilinear_crops]
    
    generated_crops = crop_images(generated, crop_locations, crop_size)
    generated_crops = [(image * 128 + 128).to(torch.uint8)
           for image in generated_crops]
    
    large_crops = crop_images(large, crop_locations, crop_size)
    large_crops = [(image * 128 + 128).to(torch.uint8)
           for image in large_crops]
    
    for crop_idx, image in enumerate(generated_crops):
        write_png(image, f'savedimages/{str(index)}/{label}_generated_part{str(crop_idx)}.png')

    if save_all:
        for crop_idx, image in enumerate(small_crops):
            write_png(image, f'savedimages/{str(index)}/small_part{str(crop_idx)}.png')
        for crop_idx, image in enumerate(bilinear_crops):
            write_png(image, f'savedimages/{str(index)}/bilinear_part{str(crop_idx)}.png')
        for crop_idx, image in enumerate(large_crops):
            write_png(image, f'savedimages/{str(index)}/large_part{str(crop_idx)}.png')

