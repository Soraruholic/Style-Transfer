# 2022.07.01
import cv2
import time
import os
import numpy as np
import random 
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets

IMAGENET_MEAN_255 = [103.53, 116.28, 123.675]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def fixRandomSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def transformImage (img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    return transform

def computeGram (tensor):
    # Compute the Gram matrix of the tensor, as well as the linear kernel matrix of the vector
    B, C, H, W = tensor.shape
    v = tensor.view (B, C, H * W)
    v_T = v.transpose (1, 2)
    G = torch.bmm (v, v_T) / (C * H * W)
    return G

def loadData (dataset_dir, img_transform, batch_size):
    train_dataset = datasets.ImageFolder(dataset_dir, transform = img_transform)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    return data_loader

def loadImage (img_path):
    # Reading images through openCV, and the image in encoded in numpy array
    img = cv2.imread (img_path)   # BGR
    return img       # RGB

def imgToTensor (img):
    # Transform the cv2 (np array) form into PIL and then into Tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])
    tensor = transform(img)

    # Unsqueeze for the batch_size dimension
    tensor = tensor.unsqueeze(dim = 0)
    return tensor

def getStyleFeatures (device, style_img_path, batch_size, VGG):
    # Load the style image
    style_img = loadImage(style_img_path)

    # Transfer the style image into tensor
    style_tensor = imgToTensor(style_img).to(device)

    # Copy the style tensor along the batc_size dimension
    B, C, H, W = style_tensor.shape

    # Compute the style features through a pretrained VGG Network
    style_representation = VGG(style_tensor.expand([batch_size, C, H, W]))

    # Compute the gram matrix 
    style_features = {}
    for key, value in style_representation.items():
        style_features[key] = computeGram(value)
        #print (key)
    return style_features 

def computeTrainingLoss (content_weight, style_weight, content_features, transformed_features, style_features, batch_size, loss):
    # Compute the content loss with MSE method
    content_loss = content_weight * loss (transformed_features['relu2_2'], content_features['relu2_2'])

    # Compute the style loss
    style_loss = 0.0
    for k, v in transformed_features.items():
        #print (style_features)
        tmp = loss (computeGram(v), style_features[k][:batch_size])
        style_loss += tmp
    style_loss *= style_weight

    # Compute the total loss
    total_loss = content_loss + style_loss

    return content_loss, style_loss, total_loss

def printLosses (num_steps, epochs, length, accumulated_content_loss, accumulated_style_loss, accumulated_total_loss, start_time):
    print(f"========Iteration {num_steps}/{epochs * length}========")
    print(f"\tContent Loss:\t{accumulated_content_loss / num_steps:.2f}")
    print(f"\tStyle Loss:\t{accumulated_style_loss / num_steps:.2f}")
    print(f"\tTotal Loss:\t{accumulated_total_loss / num_steps:.2f}")
    print(f"Time elapsed:\t{time.time() - start_time} seconds")

def saveModel (path, num_step, model):
    checkpoint = path + str (num_step) + '.pth'
    torch.save (model.state_dict (), checkpoint)
    print(f"Saved Network checkpoint file at {checkpoint}")

def tensorToImg (tensor):
    tensor = tensor.squeeze()
    img = tensor.cpu().numpy()
    img = img.transpose (1, 2, 0)
    return img

def saveImg (img, path):
    img = img.clip (0, 255)
    cv2.imwrite (path, img)

def saveTrainingSampleImage (fold, path, num_step):
    sample_tensor = fold[0].clone().detach().unsqueeze (dim = 0)
    sample_img = tensorToImg (sample_tensor.clone().detach())
    sample_path = path + 'sample_' + str (num_step) + '.jpg'
    saveImg (sample_img, sample_path)
    print(f"Saved sample tranformed image at {sample_path}")

def printSummary (network_name, start_time, stop_time, content_loss_history, style_loss_history, total_loss_history):
    print("Done Training the "+ network_name + " Network!")
    print(f"Training Time Costs: {stop_time - start_time} seconds")
    print("========Content Loss========")
    print(list(content_loss_history))
    print("========Style Loss========")
    print(list(style_loss_history))
    print("========Total Loss========")
    print(list(total_loss_history))

def saveTransformModel (model, path):
    model.eval()
    model.cpu()
    path = os.path.join ('./transform_weight', path + '.pth')
    print(f"Saving TransformerNetwork weight at {path}")
    torch.save(model.state_dict(), path)
    print("Done saving final model")


def saveLossAsCsv (content_loss_history, style_loss_history, total_loss_history):
    for index in range (len (content_loss_history)):
        content_loss_history[index] = content_loss_history[index].cpu().detach().numpy()
        style_loss_history[index] = style_loss_history[index].cpu().detach().numpy()
        total_loss_history[index] = total_loss_history[index].cpu().detach().numpy()
    ct_loss = pd.DataFrame ({'loss' : content_loss_history})
    st_loss = pd.DataFrame ({'loss' : style_loss_history})
    tt_loss = pd.DataFrame ({'loss' : total_loss_history})
    ct_loss.to_csv ('./losses/ct_loss.csv')
    st_loss.to_csv ('./losses/st_loss.csv')
    tt_loss.to_csv ('./losses/tt_loss.csv')

def show(img):
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # imshow() only accepts float [0,1] or int [0,255]
    img = np.array(img / 255).clip(0, 1)

    plt.figure(figsize = (10, 5))
    plt.imshow(img)
    plt.show()


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset for includes image file paths.
    Extends torchvision.datasets.ImageFolder()
    Reference: https://discuss.pytorch.org/t/dataloader-filenames-in-each-batch/4212/2
    """
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]

        # make a new tuple that includes original and the path
        tuple_with_path = (*original_tuple, path)
        return tuple_with_path