import os
import torch
import random
from PIL import Image
import torch.nn.functional as F

class ConvolutionFilters:
    @staticmethod
    def sobel_filler(type_approach="horizontal"):
        if type_approach == "horizontal":
            print("Using horizontal Sobel filter!")
            return torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=torch.float32)
        elif type_approach == "vertical":
            print("Using vertical Sobel filter!")
            return torch.tensor([
                [-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]
            ], dtype=torch.float32)
        else:
            raise ValueError("Invalid type_approach. Use 'horizontal' or 'vertical'.")

    @staticmethod
    def Gaussian_Blur():
        print("Using Gaussian Blur filter!")
        return torch.tensor([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=torch.float32)

    @staticmethod
    def Box_blur():
        print("Using Box Blur filter!")
        return torch.tensor([
            [.11, .11, .11],
            [.11, .11, .11],
            [.11, .11, .11]
        ], dtype=torch.float32)

    @staticmethod
    def horizontal():
        print("Using horizontal edge detection filter!")
        return torch.tensor([
            [-1,  2, -1],
            [-1,  2, -1],
            [-1,  2, -1]
        ], dtype=torch.float32)

    @staticmethod
    def vertical():
        print("Using vertical edge detection filter!")
        return torch.tensor([
            [-1, -1, -1],
            [ 2,  2,  2],
            [-1, -1, -1]
        ], dtype=torch.float32)

    @staticmethod
    def diagonal_ul_lr():
        print("Using diagonal edge detection filter (upper-left to lower-right)!")
        return torch.tensor([
            [ 2, -1, -1],
            [-1,  2, -1],
            [-1, -1,  2]
        ], dtype=torch.float32)

    @staticmethod
    def diagonal_ur_ll():
        print("Using diagonal edge detection filter (upper-right to lower-left)!")
        return torch.tensor([
            [-1, -1,  2],
            [-1,  2, -1],
            [ 2, -1, -1]
        ], dtype=torch.float32)


all_filters = {
    "sobel_horizontal": ConvolutionFilters.sobel_filler("horizontal"),
    "sobel_vertical": ConvolutionFilters.sobel_filler("vertical"),
    "gaussian_blur": ConvolutionFilters.Gaussian_Blur(),
    "box_blur": ConvolutionFilters.Box_blur(),
    "horizontal": ConvolutionFilters.horizontal(),
    "vertical": ConvolutionFilters.vertical(),
    "diagonal_ul_lr": ConvolutionFilters.diagonal_ul_lr(),
    "diagonal_ur_ll": ConvolutionFilters.diagonal_ur_ll()
}

def convolve3cTensor(inputTensor, kernel, output_image_path=None):

    C, H, W = inputTensor.shape
    kH, kW = kernel.shape

    pad_h = kH // 2
    pad_w = kW // 2

    # Pad each channel
    padded = F.pad(inputTensor, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

    output = torch.zeros((C, H, W), dtype=torch.float32, device=inputTensor.device)

    for c in range(C):
        for i in range(H):
            for j in range(W):
                region = padded[c, i:i + kH, j:j + kW]
                output[c, i, j] = torch.sum(region * kernel)

    if output_image_path != None:
        output_image = output.permute(1, 2, 0).cpu().numpy()
        output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255
        output_image = output_image.astype('uint8')
        Image.fromarray(output_image).save(output_image_path)
    return output