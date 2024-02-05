import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset

def load_single_image(directory, image_number, small_image_size, upscaling_factor):

    large_image_size = small_image_size  * upscaling_factor
    
    files = sorted(os.listdir(directory))

    full_res_image = read_image(directory + files[image_number]).to(torch.int)
    full_res_image = (full_res_image - 128) / 128                            
    large_image = transforms.Resize((large_image_size,large_image_size), 
                                    antialias = False)(full_res_image)
    small_image = transforms.Resize((small_image_size, small_image_size), 
                                    antialias = False)(full_res_image)
    
    return small_image, large_image

class ImageDataset(Dataset):

    def __init__(self, directory, small_image_size, upscaling_factor):

        super().__init__()
        self.directory = directory
        self.files = os.listdir(directory)
        self.small_image_size = small_image_size
        self.large_image_size = small_image_size * upscaling_factor

    def __len__(self):
        
        return len(self.files)

    def __getitem__(self, index):

        full_res_image = (read_image(self.directory + '/' + self.files[index])).to(torch.int)
        full_res_image = (full_res_image - 128) / 128
        large_image = transforms.Resize((self.large_image_size,self.large_image_size),
                                        antialias = False)(full_res_image)
        large_image = transforms.RandomVerticalFlip()(large_image)
        large_image = transforms.RandomHorizontalFlip()(large_image)
        small_image = transforms.Resize((self.small_image_size, self.small_image_size),
                                        antialias = False)(large_image)
        
        return small_image, large_image

def compare_model(generator, dataset_dir, small_image_size, upscaling_factor, device):

    fig, ax = plt.subplots(16, 4, figsize=(16,64))
    
    for index, axis in enumerate(ax):
        small, large = load_single_image(dataset_dir, index, small_image_size, upscaling_factor)
        
        bilinear = transforms.functional.resize(small, small_image_size * upscaling_factor, antialias=True)
        generated = generator(torch.unsqueeze(small,0).to(device)).to('cpu')
        generated = torch.movedim(torch.squeeze(generated),0,-1)
        generated = generated.detach()
        
        small = (torch.movedim(small,0,-1) * 128 + 128).to(torch.int)
        large = (torch.movedim(large,0,-1) * 128 + 128).to(torch.int)
        bilinear = (torch.movedim(bilinear,0,-1) * 128 + 128).to(torch.int)
        generated = ((generated * 128) + 128).to(torch.int)
        
        axis[0].imshow(small)
        axis[0].set_title('Small')
        axis[1].imshow(bilinear)
        axis[1].set_title('Bilinear')
        axis[2].imshow(generated)
        axis[2].set_title('Neural Net')
        axis[3].imshow(large)
        axis[3].set_title('Original')
    
        for image in axis:
            image.set_xticks([])
            image.set_yticks([])

    return fig