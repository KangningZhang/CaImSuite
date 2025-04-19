import scipy.io as sio
import numpy as np
import time
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from skimage import io
import torch
import torch.nn as nn
import torch.optim as optim
import imageio
import os
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage import generate_binary_structure
from scipy.io import savemat
from tqdm import tqdm
from scipy.io import loadmat
from matplotlib.colors import ListedColormap
from matplotlib import colors
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

def get_closest_factors(N):
    # Handle edge cases
    if N <= 0:
        return None
    if N == 1:
        return (1, 1)
    
    # Find all factors
    factors = []
    for i in range(1, int(N ** 0.5) + 1):
        if N % i == 0:
            factors.append((i, N // i))
    
    # Find the pair with minimum difference
    min_diff = float('inf')
    result = factors[0]
    
    for pair in factors:
        diff = abs(pair[0] - pair[1])
        if diff < min_diff:
            min_diff = diff
            result = pair
    
    return result


start_time = time.time()
#filename = filedialog.askopenfilename() # show an "Open" dialog box and return the path to the selected file
filename = "E:/Kangning Zhang/CaImSuite/Dataset/1P NAOMi1p v0/mov_w_bg_600_600_150_depth_100_1_512.tiff"
print(filename)

im = io.imread(filename)
im = np.float32(im)
im = im[0:im.shape[0]//2*2,:,:]
print('im shape',im.shape)
im = (im-np.min(im))/(np.max(im)-np.min(im))


im_preprocessed = im.copy() 
#im_preprocessed = np.abs((im_preprocessed - np.mean(im_preprocessed, axis=0)))
#im_preprocessed = (im_preprocessed-np.min(im_preprocessed))/(np.max(im_preprocessed)-np.min(im_preprocessed))


# After creating timestamp and directories
timestamp = time.strftime("%Y%m%d_%H%M%S")
base_filename = os.path.basename(filename).rsplit('.', 1)[0]  # Get filename without extension
base_save_dir = f'./Results_{base_filename}_{timestamp}'
figures_dir = os.path.join(base_save_dir, 'Figures')
model_dir = os.path.join(base_save_dir, 'Models')
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
print(f"Results will be saved in: {base_save_dir}")

# Modify GIF saving code
# Generate output filenames
base_filename = os.path.basename(filename).rsplit('.', 1)[0]
class CaImSuite(nn.Module):
    def __init__(self, dim_x, dim_y, dim_t):
        super(CaImSuite, self).__init__()
        self.N = 32
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_t = dim_t
        
        # Define ISTA block layers
        self.conv1 = nn.Conv3d(1, self.N, kernel_size=(1,3,3), padding='same')
        self.conv2 = nn.Conv3d(self.N, self.N, kernel_size=(1,3,3), padding='same')
        self.conv3 = nn.Conv3d(self.N, self.N, kernel_size=(1,3,3), padding='same')
        self.conv4 = nn.Conv3d(self.N, self.N, kernel_size=(1,3,3), padding='same')
        self.conv5 = nn.Conv3d(self.N, 1, kernel_size=(1,3,3), padding='same')
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def ISTA_blocks(self, R):
        temp = 0.01
        
        conv1 = self.conv1(R)
        conv2 = self.relu(self.conv2(conv1))
        conv_symm = self.conv2(conv1)
        
        conv3 = self.conv3(conv2)
        conv_symm = self.conv3(conv_symm)
        
        # Soft thresholding
        conv4 = torch.sign(conv3) * self.relu(torch.abs(conv3) - temp)
        
        conv5 = self.relu(self.conv4(conv4))
        conv_symm = self.relu(self.conv4(conv_symm))
        
        conv6 = self.conv5(conv5)
        conv_symm = self.conv5(conv_symm)
        
        conv7 = conv6 + R
        conv8 = conv_symm - conv1
        
        return conv4, conv7, conv8

    def ISTA_Siamese(self, x):
        temp = 0.01
        
        # First block
        R1 = x
        _, conv1_7, conv1_8 = self.ISTA_blocks(R1)
        
        # Subsequent blocks
        R2 = conv1_7 - temp * (conv1_7 - x)
        _, conv2_7, conv2_8 = self.ISTA_blocks(R2)
        
        R3 = conv2_7 - temp * (conv2_7 - x)
        _, conv3_7, conv3_8 = self.ISTA_blocks(R3)
        
        R4 = conv3_7 - temp * (conv3_7 - x)
        _, conv4_7, conv4_8 = self.ISTA_blocks(R4)
        
        R5 = conv4_7 - temp * (conv4_7 - x)
        conv5_4, Denoise, conv5_8 = self.ISTA_blocks(R5)
        
        # Combine symmetric outputs
        output_symm = conv1_8 + conv2_8 + conv3_8 + conv4_8 + conv5_8
        output_symm = torch.mean(output_symm, dim=1, keepdim=True) / 5
        
        Sparse_prepro = torch.abs(conv5_4)
        Sparse_prepro = self.tanh(Sparse_prepro)
        
        return Denoise, Sparse_prepro, output_symm

    def forward(self, x):
        Denoise, Sparse_prepro, output_symm = self.ISTA_Siamese(x)
        return torch.cat([Denoise, Sparse_prepro, output_symm], dim=1)

def denoise_loss(pred, target):
    mse_loss = torch.mean((pred[:,:1] - target[:,:1])**2)
    reg_loss = 0.01 * torch.mean(pred[:,-1:]**2)
    return mse_loss + reg_loss

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_epoch = 1
batch_size = 1

# Prepare data
data = torch.from_numpy(im_preprocessed[0:im_preprocessed.shape[0]:2]).float()
target = torch.from_numpy(im_preprocessed[1:im_preprocessed.shape[0]:2]).float()
N_iteration = int(data.shape[0]/batch_size)

print('data shape',data.shape)
print('target shape',target.shape)
# Initialize model
model = CaImSuite(data.shape[1], data.shape[2], data.shape[0]).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)

loss_history = []
start_time_total = time.time()

for epoch in range(N_epoch):
    indices = torch.randperm(data.shape[0])
    start_time = time.time()
    
    for iteration in range(N_iteration):
        idx = indices[iteration*batch_size:(iteration+1)*batch_size]
        
        # Prepare batch data
        data_batch = torch.stack([
            torch.unsqueeze(data[i:i+1], 0)
            for i in idx
        ]).to(device)
        
        target_batch = torch.stack([
            torch.unsqueeze(target[i:i+1], 0)
            for i in idx
        ]).to(device)
        # Training step
        optimizer.zero_grad()
        output = model(data_batch)

        loss = denoise_loss(output, target_batch)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        print(f'Epoch: {epoch+1}/{N_epoch} Iteration: {iteration+1}/{N_iteration}, Loss_denoise: {loss_val:.6f}')
        
        # Visualization code (every 20% of iterations)
        if (iteration+1) % int(0.2*data.shape[0]/batch_size) == 0:
            with torch.no_grad():
                img = model(data_batch[0:1].to(device))
                # Squeeze out extra dimensions
                img_denoised = img[0,0].cpu().numpy().squeeze()
                img_sparse = img[0,1:1+model.N].mean(dim=0).cpu().numpy().squeeze()
                img_raw = data_batch[0,0].cpu().numpy().squeeze()
                img_gt = target_batch[0,0].cpu().numpy().squeeze()
                
                plt.figure(figsize=(20,5))
                plt.subplot(1,4,1)
                plt.imshow(img_raw, vmin=0, vmax=0.5, cmap='gray')
                plt.axis('off')
                plt.title('Raw')
                plt.subplot(1,4,2)
                plt.imshow(img_denoised, vmin=0, vmax=0.5, cmap='gray')
                plt.axis('off')
                plt.title('Denoise')
                plt.subplot(1,4,3)
                plt.imshow(img_sparse, cmap='gray')
                plt.axis('off')
                plt.title('Seg')
                plt.subplot(1,4,4)
                plt.imshow(img_gt, vmin=0, vmax=0.5, cmap='gray')
                plt.axis('off')
                plt.title('Denoise GT')
                
                # Save figure
                fig_filename = os.path.join(figures_dir, f'epoch_{epoch+1}_iter_{iteration+1}.png')
                plt.savefig(fig_filename, bbox_inches='tight', dpi=300)
                
                print(f"Figure saved as: {fig_filename}")
                
                # Save model
                model_path = os.path.join(model_dir, f'CaImSuite_model_epoch_{epoch+1}_iter_{iteration+1}.pth')
                torch.save(model.state_dict(), model_path)
                print(f"Model saved as: {model_path}")
        
        print(f"--- {time.time() - start_time:.2f} seconds elapsed ---")
    
    print(f"\x1b[31m--- {(time.time() - start_time)/60:.2f} minutes elapsed for this epoch ---\x1b[0m")
    print()

print(f"\x1b[31m--- {(time.time() - start_time_total)/3600:.2f} hours elapsed for training ---\x1b[0m")

# Add after training loop
print("\nStarting validation on preprocessed images...")

# Create validation results directory
val_dir = os.path.join(base_save_dir, 'Validation')
os.makedirs(val_dir, exist_ok=True)

# Convert preprocessed images to torch tensor
val_data = torch.from_numpy(im_preprocessed).float().unsqueeze(1).to(device)  # Add channel dimension
val_data = torch.unsqueeze(val_data, 1)

# Process in smaller batches to avoid memory issues
batch_size_val = 1
val_results = []
model.eval()

with torch.no_grad():
    for i in tqdm(range(0, len(val_data), batch_size_val)):
        batch = val_data[i:i+batch_size_val]
        outputs = model(batch)
        val_results.append(outputs.cpu().numpy())

val_results = np.concatenate(val_results, axis=0)
print('val_results shape',val_results.shape)

# Save validation results
np.save(os.path.join(val_dir, 'val_results.npy'), val_results)
print(f"Validation results saved to: {os.path.join(val_dir, 'val_results.npy')}")

K = 4
n_channels = K

# Save sparse representation results
sparse_maps = val_results[:, 1:1+model.N]  # Get sparse maps
for i in range(sparse_maps.shape[1]):
    channel = sparse_maps[:, i]
    channel_median = np.median(channel)
    channel_max = np.max(channel)
    temp= (channel - channel_median) / (channel_max - channel_median + 1e-10)
    temp[temp<0] = 0
    sparse_maps[:, i] = temp

# Plot all channels before filtering
original_n_channels = model.N
n_subplot = int(np.ceil(np.sqrt(original_n_channels)))**2
subplot_dim = int(np.sqrt(n_subplot))

# Get image dimensions from the original data
im_shape = im_preprocessed.shape[1:]  # Get height and width from preprocessed image

# Filter channels based on max/median ratio and high-intensity pixels
valid_channels = []
filtered_sparse_maps = []
for i in range(sparse_maps.shape[1]):
    channel = sparse_maps[:, i]
    channel_max = np.squeeze(np.max(channel, axis=0))
    max_val = np.max(channel_max)
    min_val = np.min(channel_max)
    
    # Count pixels above 30% of max value
    threshold = 0.5 * max_val
    high_intensity_pixels = np.sum(channel_max > threshold)
    valid_channels.append((i, high_intensity_pixels))

sparse_maps = np.squeeze(sparse_maps)

if valid_channels:
    valid_channels.sort(key=lambda x: x[1], reverse=True)  # Sort by pixel count
    valid_channels = [x[0] for x in valid_channels]
    filtered_sparse_maps = [sparse_maps[:, i] for i in valid_channels]
    sparse_maps_copy = np.stack(filtered_sparse_maps, axis=1)

plt.figure(figsize=(15, 15))
for i in range(subplot_dim):
    for j in range(subplot_dim):
        idx = i*subplot_dim + j
        if idx < original_n_channels:
            channel = sparse_maps_copy[:, idx]
            channel_max = np.squeeze(np.max(channel, axis=0))
            max_val = np.max(channel_max)
            min_val = np.min(channel_max)
            threshold = 0.5 * max_val
            high_intensity_pixels = np.sum(channel_max > threshold)
            plt.subplot(subplot_dim, subplot_dim, idx+1)
            
            plt.imshow(channel_max, cmap='gray', vmin=0, vmax=0.5)
            plt.axis('off')
            plt.title(f'channel {idx+1}, N: {high_intensity_pixels}')
plt.suptitle('All Channels Before Filtering', y=0.92)
plt.savefig(os.path.join(val_dir, 'All_Channels_Before_Filtering.jpg'), dpi=300)
plt.close()
print('All channels before filtering saved')

filtered_sparse_maps = []
# Sort channels by number of high-intensity pixels and keep top 10
if valid_channels:
    filtered_sparse_maps = [sparse_maps_copy[:, i] for i in range(K)]
    sparse_maps = np.stack(filtered_sparse_maps, axis=1)
    
    print(f"Kept {K} channels out of {model.N} channels")
else:
    print("Warning: No channels met the filtering criterion")
    # Keep original sparse_maps if no channels pass the filter

def histogram_cluster_frame(feature_maps, frame_idx, val_dir):
    """
    Perform pixelwise histogram clustering on feature maps
    feature_maps shape: (C, H, W) where C is number of channels
    """
    C, H, W = feature_maps.shape
    
    # Reshape to (H*W, C) for feature analysis
    features = feature_maps.transpose(1, 2, 0).reshape(-1, C)
    
    # Compute feature intensity (mean across channels for each pixel)
    pixel_intensities = np.mean(features, axis=1)
    print('pixel_intensities shape',pixel_intensities.shape)
    
    # Estimate density using KDE
    kde = gaussian_kde(pixel_intensities)
    x_grid = np.linspace(pixel_intensities.min(), pixel_intensities.max(), 200)
    density = kde(x_grid)
    plt.plot(x_grid, density)
    plt.savefig(f'{val_dir}/Density_frame_{frame_idx}.jpg', dpi=300)
    plt.close()
    
    # Find valleys in density (separation points)
    peaks, _ = find_peaks(-density)  # Finding valleys by inverting density
    
    if len(peaks) > 0:
        # Use the most prominent valley as threshold
        threshold = x_grid[peaks[np.argmax(-density[peaks])]]
    else:
        # If no clear valley, use median as threshold
        threshold = np.median(pixel_intensities)
    
    # Create binary segmentation
    binary_seg = (pixel_intensities > threshold).reshape(H, W)
    
    return binary_seg.astype(float)

# Replace simple thresholding with pixelwise histogram clustering
print("Performing pixelwise histogram clustering...")
seg = np.zeros_like(sparse_maps[:, 0, :, :])  # Initialize segmentation array
for frame_idx in tqdm(range(sparse_maps.shape[0])):
    frame_features = sparse_maps[frame_idx]  # Shape: (C, H, W)
    seg[frame_idx] = histogram_cluster_frame(frame_features, frame_idx, val_dir)

# Create figure for max projection
plt.figure(figsize=(10, 8))
plt.subplot(111)
plt.imshow(np.squeeze(np.max(seg, axis=0)),cmap='gray',vmin=0,vmax=0.5)
plt.axis('off')
plt.colorbar()
plt.savefig(val_dir+'/Segmentation.jpg', dpi=300)
plt.close()
print('Segmentation saved')

# Create figure for max projection
plt.figure(figsize=(10, 8))
plt.subplot(111)
plt.imshow(np.squeeze(np.max(np.mean(sparse_maps,axis=1), axis=0)),cmap='gray',vmin=0,vmax=0.5)
plt.axis('off')
plt.colorbar()
plt.savefig(val_dir+'/Sparse_Projection.jpg', dpi=300)
plt.close()
print('Sparse Projection saved')

# Update the sparse representation visualization to only show valid channels

n_subplot = int(np.ceil(np.sqrt(n_channels)))**2  # Get closest greater square number
subplot_dim = int(np.sqrt(n_subplot))  # Get dimension for square grid

plt.figure(figsize=(10, 8))
for i in range(subplot_dim):
    for j in range(subplot_dim):
        idx = i*subplot_dim + j
        if idx < K:
            plt.subplot(subplot_dim, subplot_dim, idx+1)
            plt.imshow(np.squeeze(np.max(sparse_maps[:,idx], axis=0)), cmap='gray',vmin=0,vmax=0.5)
            plt.axis('off')
            plt.title(f'channel {idx+1}')
plt.savefig(val_dir+'/Sparse_representation.jpg', dpi=300)
plt.close()
print('Sparse representation saved')

# Create figure for denoised projection
plt.figure(figsize=(10, 8))
plt.subplot(111)
plt.imshow(np.squeeze(np.max(im_preprocessed, axis=0)),cmap='gray',vmin=0,vmax=0.5)
plt.axis('off')
plt.colorbar()
plt.title('Denoised Projection')
plt.savefig(val_dir+'/Denoised.jpg')
plt.close()
print('Denoised Projection saved')
# Create figure for raw projection
filename = "E:/Kangning Zhang/CaImSuite/Dataset/1P NAOMi1p v0/mov_w_bg_600_600_150_depth_100_1_512.tiff"
im = io.imread(filename)
im = np.float32(im)
im = im[0:im.shape[0]//2*2,:,:]
im = (im-np.min(im))/(np.max(im)-np.min(im))
plt.figure(figsize=(10, 8))
plt.subplot(111)
plt.imshow(np.squeeze(np.max(im, axis=0)),cmap='gray',vmin=0,vmax=0.5)
plt.axis('off')
plt.colorbar()
plt.title('Raw Projection')
plt.savefig(val_dir+'/Raw.jpg', dpi=300)
plt.close()
print('Raw Projection saved')

# Load ground truth mask
gt_mask = loadmat(os.path.join(os.path.dirname(filename), 'mask_video_1.mat'))['mask_filtered']
gt_max = np.squeeze(np.max(gt_mask, axis=2))

# Create figure for overlay comparison
plt.figure(figsize=(10, 8))

# Get denoised max projection and enhance contrast
denoised_max = np.squeeze(np.max(im_preprocessed, axis=0))
vmin = np.percentile(denoised_max, 1)
vmax = np.percentile(denoised_max, 99)

# Create binary structure for edge detection
struct = generate_binary_structure(2, 2)

# Get contours for both segmentations
seg_max = np.squeeze(np.max(seg, axis=0))
seg_contours = binary_dilation(seg_max, struct) & ~binary_erosion(seg_max, struct)
gt_contours = binary_dilation(gt_max, struct) & ~binary_erosion(gt_max, struct)

# Plot base denoised image
im = plt.imshow(denoised_max, cmap='gray', vmin=0, vmax=0.5)
plt.colorbar(im)

# Add ground truth contours in green
gt_overlay = plt.imshow(gt_contours, cmap=ListedColormap(['none', 'green']), alpha=1, vmin=0, vmax=0.5)

# Add our segmentation contours in red
seg_overlay = plt.imshow(seg_contours, cmap=ListedColormap(['none', 'red']), alpha=1, vmin=0, vmax=0.5)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=1, label='Ground Truth'),
    Patch(facecolor='red', alpha=1, label='Our Segmentation')
]
plt.legend(handles=legend_elements, loc='lower left')

plt.axis('off')
plt.title('Segmentation Comparison overlayed on max projection of image without background')

plt.tight_layout()
plt.savefig(os.path.join(val_dir, 'Segmentation_Comparison.jpg'), dpi=300)
plt.close()
print('Segmentation Comparison saved')
print('Total time elapsed (seconds) %4.2f: ',time.time()-start_time_total)


