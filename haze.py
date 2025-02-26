import cv2
import numpy as np
import os
import torch
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from monodepth import networks
from torchvision import transforms
from monodepth.layers import disp_to_depth

import shutil
import json
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import warnings
from torchvision.models import _utils


warnings.filterwarnings("ignore", category=UserWarning, module=_utils.__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=Image.__name__)

# Add missing import for Image.Resampling
from PIL import Image

def create_depth_map(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./models/mono+stereo_1024x320"
    # print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    # print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    # print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
    with torch.no_grad():

        input_image = pil.open(file_path).convert('RGB')
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), Image.Resampling.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # Prediction
        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

        # Convert the depth map to a numpy array
        depth_map = disp_resized.squeeze().cpu().numpy()

        return depth_map


def add_haze_to_image(image_file, depth_map=None, k=0.5, beta=0.00008, display=False):
    """
    Adds haze to an image using either an existing depth map or a generated depth map.

    Args:
        image_file (str): Path to the original image
        depth_map (str, optional): Path to the depth map if available
        k (float): Atmospheric light contribution
        beta (float): Haze attenuation coefficient
        display (bool): Whether to display images before conversion (default: False)

    Returns:
        PIL Image: Hazy image with simulated fog.
    """

    # Load the image
    image = Image.open(image_file).convert("RGB")
    image_np = np.array(image, dtype=np.float32)


    ### 💡 **Haze Computation (Previously `generate_haze`)**
    haze_k = k + 0.3  # Adjusted atmospheric light contribution
    haze_beta = beta  # Attenuation coefficient controlling haze density

    transmitmap = np.expand_dims(np.exp(-1 * haze_beta * depth_map), axis=2)
    tx = np.concatenate([transmitmap, transmitmap, transmitmap], axis=2)
    txcvt = (tx * 255).astype('uint8')

    # Apply guided filter to smooth the transmission map
    tx_filtered = cv2.ximgproc.guidedFilter(guide=image_np.astype('uint8'), src=txcvt, radius=50, eps=1e-3, dDepth=-1)

    # Apply haze model equation
    fog_image = (image_np / 255) * (tx_filtered / 255) + haze_k * (1 - tx_filtered / 255)
    fog_image = np.clip(fog_image, 0, 1)

    hazy_image = (fog_image * 255).astype('uint8')

    # Display if required
    if display:
        image.show(title='Original Image')
        Image.fromarray(hazy_image).show(title='Hazy Image')

    return Image.fromarray(hazy_image)



def display_kitti15_samples(directory, num_samples=2):
    samples_processed = 0
    for root, _, files in os.walk(directory):
        if not os.path.basename(root) in ['image_2', 'image_3']:
            continue

        for file in files:
            if samples_processed >= num_samples:
                break

            file_path = os.path.join(root, file)

            if file_path.endswith('.png'):
                print(f"Displaying sample {samples_processed + 1}: {file_path}")
                depth_map = create_depth_map(file_path)
                hazy_image = add_haze_to_image(file_path, depth_map, display=False)  # Set display to False to avoid showing the image twice

                # Display the original, hazy image and depth map
                original_image = cv2.imread(file_path)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                depth_colormap = cm.jet(depth_map)
                
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(original_image)
                ax[0].set_title('Original Image')
                ax[0].axis('off')
                
                ax[1].imshow(np.asarray(hazy_image))
                ax[1].set_title('Hazy Image')
                ax[1].axis('off')

                ax[2].imshow(depth_colormap)
                ax[2].set_title('Depth Map')
                ax[2].axis('off')
                
                plt.show()

                samples_processed += 1

        if samples_processed >= num_samples:
            break

    print(f"Displayed kitti15 {samples_processed} samples")

def display_sintel_samples(input_dir, num_samples=2):
    samples_processed = 0
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if samples_processed >= num_samples:
                break

            if not filename.endswith(".png"):
                continue

            if "flow" in dirpath or "invalid" in dirpath:
                continue

            file_path = os.path.join(dirpath, filename)

            print(f"Displaying sample {samples_processed + 1}: {file_path}")
            depth_map = create_depth_map(file_path)
            hazy_image = add_haze_to_image(file_path, depth_map, display=False)  # Set display to False to avoid showing the image twice

            # Display the original, hazy image and depth map
            original_image = cv2.imread(file_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            depth_colormap = cm.jet(depth_map)
            
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(original_image)
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            
            ax[1].imshow(np.asarray(hazy_image))
            ax[1].set_title('Hazy Image')
            ax[1].axis('off')

            ax[2].imshow(depth_colormap)
            ax[2].set_title('Depth Map')
            ax[2].axis('off')
            
            plt.show()

            samples_processed += 1

        if samples_processed >= num_samples:
            break

    print(f"Displayed sintel {samples_processed} samples")


def display_flyingchairs_samples(input_dir, num_samples=2):
    samples_processed = 0
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if samples_processed >= num_samples:
                break

            if not filename.endswith(".ppm"):
                continue

            if "flow" in filename:
                continue

            file_path = os.path.join(dirpath, filename)

            print(f"Displaying sample {samples_processed + 1}: {file_path}")
            depth_map = create_depth_map(file_path)
            hazy_image = add_haze_to_image(file_path, depth_map, display=False)  # Set display to False to avoid showing the image twice

            # Display the original, hazy image and depth map
            original_image = cv2.imread(file_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            depth_colormap = cm.jet(depth_map)
            
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(original_image)
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            
            ax[1].imshow(np.asarray(hazy_image))
            ax[1].set_title('Hazy Image')
            ax[1].axis('off')

            ax[2].imshow(depth_colormap)
            ax[2].set_title('Depth Map')
            ax[2].axis('off')
            
            plt.show()

            samples_processed += 1

        if samples_processed >= num_samples:
            break

    print(f"Displayed flyingchairs {samples_processed} samples")

def display_flyingthings3d_samples(directory, num_samples=2):
    samples_processed = 0
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if samples_processed >= num_samples:
                break

            if not filename.endswith(".png"):
                continue

            if not os.path.basename(dirpath) in ['left', 'right']:
                continue

            file_path = os.path.join(dirpath, filename)

            print(f"Displaying sample {samples_processed + 1}: {file_path}")
            depth_map = create_depth_map(file_path)
            hazy_image = add_haze_to_image(file_path, depth_map, display=False)  # Set display to False to avoid showing the image twice

            # Display the original, hazy image and depth map
            original_image = cv2.imread(file_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            depth_colormap = cm.jet(depth_map)
            
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(original_image)
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            
            ax[1].imshow(np.asarray(hazy_image))
            ax[1].set_title('Hazy Image')
            ax[1].axis('off')

            ax[2].imshow(depth_colormap)
            ax[2].set_title('Depth Map')
            ax[2].axis('off')
            
            plt.show()

            samples_processed += 1
        if samples_processed >= num_samples:
            break

    print(f"Displayed flyingthings3d {samples_processed} samples")

def display_hd1k_samples(directory, num_samples=2):
    samples_processed = 0
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if samples_processed >= num_samples:
                break

            if not filename.endswith(".png"):
                continue

            if 'hd1k_flow_gt' in dirpath or 'hd1k_flow_uncertainty' in dirpath:
                continue

            file_path = os.path.join(dirpath, filename)

            print(f"Displaying sample {samples_processed + 1}: {file_path}")
            depth_map = create_depth_map(file_path)
            hazy_image = add_haze_to_image(file_path, depth_map, display=False)  # Set display to False to avoid showing the image twice

            # Display the original, hazy image and depth map
            original_image = cv2.imread(file_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            depth_colormap = cm.jet(depth_map)
            
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(original_image)
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            
            ax[1].imshow(np.asarray(hazy_image))
            ax[1].set_title('Hazy Image')
            ax[1].axis('off')

            ax[2].imshow(depth_colormap)
            ax[2].set_title('Depth Map')
            ax[2].axis('off')
            
            plt.show()

            samples_processed += 1

        if samples_processed >= num_samples:
            break

    print(f"Displayed HD1K {samples_processed} samples")

def display_chairssdhom_samples(directory, num_samples=2):
    samples_processed = 0

    target_directories = [os.path.join(directory, "data", d, sd) for d in ["train", "test"] for sd in ["t0", "t1"]]

    for target_dir in target_directories:
        for dirpath, _, filenames in os.walk(target_dir):
            for filename in filenames:
                if samples_processed >= num_samples:
                    break

                if not filename.endswith(".png"):
                    continue

                file_path = os.path.join(dirpath, filename)

                print(f"Displaying sample {samples_processed + 1}: {file_path}")
                depth_map = create_depth_map(file_path)
                hazy_image = add_haze_to_image(file_path, depth_map, display=False)  # Set display to False to avoid showing the image twice

                # Display the original, hazy image and depth map
                original_image = cv2.imread(file_path)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                depth_colormap = cm.jet(depth_map)
                
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(original_image)
                ax[0].set_title('Original Image')
                ax[0].axis('off')
                
                ax[1].imshow(np.asarray(hazy_image))
                ax[1].set_title('Hazy Image')
                ax[1].axis('off')

                ax[2].imshow(depth_colormap)
                ax[2].set_title('Depth Map')
                ax[2].axis('off')
                
                plt.show()

                samples_processed += 1

            if samples_processed >= num_samples:
                break

    print(f"Displayed ChairsSDHom {samples_processed} samples")

def display_flyingchairssocc_samples(directory, num_samples=2):
    samples_processed = 0
    image_files = []

    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if not (filename.endswith("_img1.png") or filename.endswith("_img2.png")):
                continue

            file_path = os.path.join(dirpath, filename)
            print(f"Displaying sample {samples_processed + 1}: {file_path}")
            depth_map = create_depth_map(file_path)
            hazy_image = add_haze_to_image(file_path, depth_map, display=False)  # Set display to False to avoid showing the image twice
            # Display the original image
            original_image = cv2.imread(file_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            depth_colormap = cm.jet(depth_map)
            
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(original_image)
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            
            ax[1].imshow(np.asarray(hazy_image))
            ax[1].set_title('Hazy Image')
            ax[1].axis('off')

            ax[2].imshow(depth_colormap)
            ax[2].set_title('Depth Map')
            ax[2].axis('off')
            
            plt.show()
            samples_processed += 1
            if samples_processed >= num_samples:
                break
        if samples_processed >= num_samples:
            break

    print(f"Displayed FlyingChairsOcc {samples_processed} samples")

def display_flyingthings3d_subset_samples(directory, num_samples=2):
    samples_processed = 0  
    image_files = [os.path.join(dirpath, filename) for dirpath, _, filenames in os.walk(directory) for filename in filenames if filename.endswith(".png")]

    for file_path in image_files:
        if samples_processed >= num_samples:
            break

        print(f"Displaying sample {samples_processed + 1}: {file_path}")
        depth_map = create_depth_map(file_path)
        hazy_image = add_haze_to_image(file_path, depth_map, display=False)  # Set display to False to avoid showing the image twice

        # Display the original and hazy image
        original_image = cv2.imread(file_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(original_image)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(np.asarray(hazy_image))
        ax[1].set_title('Hazy Image')
        ax[1].axis('off')

        plt.show()

        samples_processed += 1

    print(f"Displayed FlyingThings3D_subset {samples_processed} samples")


def clone_directory(src, dst):
	shutil.copytree(src, dst)

def get_depth_from_disparity(image_file, focal_length=721.5377, baseline=0.54):
    """
    Retrieves and converts a KITTI disparity map to a depth map if available.

    Args:
        image_file (str): Path to the original image.
        focal_length (float, optional): Camera focal length (default: KITTI value).
        baseline (float, optional): Camera baseline distance (default: KITTI value).

    Returns:
        numpy array: Depth map if disparity exists, otherwise None.
    """
    # Determine disparity map path
    parent = os.path.dirname(image_file)
    grandparent = os.path.dirname(parent)
    folder_name = os.path.basename(parent)

    if folder_name == "image_2":
        disparity_path = os.path.join(grandparent, "disp_occ_0", os.path.basename(image_file))
    elif folder_name == "image_3":
        disparity_path = os.path.join(grandparent, "disp_occ_1", os.path.basename(image_file))
    else:
        return None  # Not a valid KITTI image folder

    # If the disparity file exists, load it and convert to depth
    if os.path.exists(disparity_path):
        disparity_map = cv2.imread(disparity_path, cv2.IMREAD_ANYDEPTH)
        if disparity_map is not None:
            disparity_map = disparity_map.astype(np.float32) / 256.0  # Normalize disparity
            depth_map = (focal_length * baseline) / (disparity_map + 1e-6)  # Convert to depth
            return depth_map
        else:
            print(f"⚠️ Warning: Could not read disparity map at {disparity_path}")
    
    return None  # Return None if disparity map doesn't exist or couldn't be read


def add_haze_to_kitti_images_in_directory(directory, checkpoint_file, k=0.5, beta=0.00008):
    """
    Processes KITTI images by adding haze, using either pre-existing disparity maps 
    (converted to depth) or generating depth maps if disparity is unavailable.

    Args:
        directory (str): Path to KITTI dataset directory.
        checkpoint_file (str): Path to the checkpoint file for tracking processed images.
        k (float, optional): Atmospheric light contribution (default: 0.5).
        beta (float, optional): Haze attenuation coefficient (default: 0.00008).

    Returns:
        int: Number of images processed.
    """

    num_images = 0
    checkpoints = {}

    # Load checkpoint file if it exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoints = json.load(f)

    # Collect image files from KITTI's left/right camera directories
    image_files = []
    for root, _, files in os.walk(directory):
        if os.path.basename(root) not in ['image_2', 'image_3']:
            continue
        for file in files:
            if file.endswith('.png'):
                image_files.append(os.path.join(root, file))

    print(f"Total number of samples: {len(image_files)}")

    start_time = time.time()

    for file_path in tqdm(image_files, desc="Adding haze"):
        rel_file_path = os.path.relpath(file_path, directory)

        # Skip already processed images
        if rel_file_path in checkpoints and checkpoints[rel_file_path] == True:
            print(f"Skipping already processed image: {file_path}")
            continue

        # Get depth map: Prefer KITTI disparity-to-depth, fallback to estimated depth
        depth_map = get_depth_from_disparity(file_path)
        if depth_map is None:
            print(f"⚠️ No disparity found for {file_path}, generating depth map.")
            depth_map = create_depth_map(file_path)

        # Apply haze transformation
        hazy_image = add_haze_to_image(file_path, depth_map, k, beta)
        hazy_image.save(file_path)  # Overwrite image with hazy version

        # Update checkpoint file
        num_images += 1
        checkpoints[rel_file_path] = True
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoints, f)

    end_time = time.time()
    print(f"✅ Added haze to {num_images} images in {end_time - start_time:.2f} seconds.")

    return num_images


##############################
#adding haze to sintel dataset

def add_haze_to_sintel_images_in_directory(directory, checkpoint_file, k=0.5, beta=0.00008, display=False):

    num_samples = 0
    checkpoints = {}

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoints = json.load(f)

    image_files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".png") and "flow" not in dirpath and "invalid" not in dirpath:
                image_files.append(os.path.join(dirpath, filename))

    print(f"Total number of samples: {len(image_files)}")

    start_time = time.time()
    for file_path in tqdm(image_files, desc="Adding haze"):
        rel_file_path = os.path.relpath(file_path, directory)
        # outpath = os.path.join(output_dir, rel_file_path)

        if rel_file_path in checkpoints and checkpoints[rel_file_path] == True:
            print(f"Skipping already processed image: {file_path}")
        else:
            depth_map = create_depth_map(file_path)
            hazy_image = add_haze_to_image(file_path, depth_map, k, beta)
            # os.makedirs(os.path.dirname(outpath), exist_ok=True)
            hazy_image.save(file_path)
            num_samples += 1
            checkpoints[rel_file_path] = True
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoints, f)

    end_time = time.time()
    print(f"Time taken to add haze: {end_time - start_time:.2f} seconds")

    return num_samples


#################
### adding haze to flyingchairs dataset

def add_haze_to_flyingchairs_images_in_directory(directory, checkpoint_file, k=0.5, beta=0.00008, display=False):

    num_samples = 0
    checkpoints = {}

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoints = json.load(f)

    image_files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".ppm"):
                image_files.append(os.path.join(dirpath, filename))

    print(f"Total number of samples: {len(image_files)}")

    start_time = time.time()
    for file_path in tqdm(image_files, desc="Adding haze"):
        rel_file_path = os.path.relpath(file_path, directory)

        if rel_file_path in checkpoints and checkpoints[rel_file_path] == True:
            print(f"Skipping already processed image: {file_path}")
        else:
            depth_map = create_depth_map(file_path)
            hazy_image = add_haze_to_image(file_path, depth_map, k, beta)
            hazy_image.save(file_path)
            num_samples += 1
            checkpoints[rel_file_path] = True
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoints, f)

    end_time = time.time()
    print(f"Time taken to add haze: {end_time - start_time:.2f} seconds")

    return num_samples

def add_haze_to_flyingthings3d_images_in_directory(directory, checkpoint_file, k=0.5, beta=0.00008, display=False):
    num_samples = 0
    checkpoints = {}

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoints = json.load(f)

    image_files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".png") and (os.path.basename(dirpath) in ['left', 'right']):
                image_files.append(os.path.join(dirpath, filename))

    print(f"Total number of samples: {len(image_files)}")

    start_time = time.time()
    for file_path in tqdm(image_files, desc="Adding haze"):
        rel_file_path = os.path.relpath(file_path, directory)

        if rel_file_path in checkpoints and checkpoints[rel_file_path] == True:
            print(f"Skipping already processed image: {file_path}")
        else:
            depth_map = create_depth_map(file_path)
            hazy_image = add_haze_to_image(file_path, depth_map, k, beta)
            hazy_image.save(file_path)
            num_samples += 1
            checkpoints[rel_file_path] = True
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoints, f)

    end_time = time.time()
    print(f"Time taken to add haze: {end_time - start_time:.2f} seconds")

    return num_samples

def add_haze_to_hd1k_images_in_directory(directory, checkpoint_file, k=0.5, beta=0.00008, display=False):
    num_samples = 0
    checkpoints = {}

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoints = json.load(f)

    image_files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".png") and 'hd1k_flow_gt' not in dirpath and 'hd1k_flow_uncertainty' not in dirpath:
                image_files.append(os.path.join(dirpath, filename))

    print(f"Total number of samples: {len(image_files)}")

    start_time = time.time()
    for file_path in tqdm(image_files, desc="Adding haze"):
        rel_file_path = os.path.relpath(file_path, directory)

        if rel_file_path in checkpoints and checkpoints[rel_file_path] == True:
            print(f"Skipping already processed image: {file_path}")
        else:
            depth_map = create_depth_map(file_path)
            hazy_image = add_haze_to_image(file_path, depth_map, k, beta)
            hazy_image.save(file_path)
            num_samples += 1
            checkpoints[rel_file_path] = True
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoints, f)

    end_time = time.time()
    print(f"Time taken to add haze: {end_time - start_time:.2f} seconds")

    return num_samples


def add_haze_to_chairssdhom_images_in_directory(directory, checkpoint_file, k=0.5, beta=0.00008, display=False):
    num_samples = 0
    checkpoints = {}

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoints = json.load(f)
            
    target_directories = [os.path.join(directory, "data", d, sd) for d in ["train", "test"] for sd in ["t0", "t1"]]
    image_files = []

    for target_dir in target_directories:
        image_files.extend([os.path.join(dirpath, filename) for dirpath, _, filenames in os.walk(target_dir) for filename in filenames if filename.endswith(".png")])

    print(f"Total number of samples: {len(image_files)}")

    start_time = time.time()
    for file_path in tqdm(image_files, desc="Adding haze"):
        rel_file_path = os.path.relpath(file_path, directory)

        if rel_file_path in checkpoints and checkpoints[rel_file_path] == True:
            print(f"Skipping already processed image: {file_path}")
        else:
            depth_map = create_depth_map(file_path)
            hazy_image = add_haze_to_image(file_path, depth_map, k, beta)
            hazy_image.save(file_path)
            num_samples += 1
            checkpoints[rel_file_path] = True
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoints, f)

    end_time = time.time()
    print(f"Time taken to add haze: {end_time - start_time:.2f} seconds")

    return num_samples

def add_haze_to_flyingchairsocc(directory, checkpoint_file, k=0.5, beta=0.00008):
    num_samples = 0
    checkpoints = {}

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoints = json.load(f)

    image_files = [os.path.join(dirpath, filename) for dirpath, _, filenames in os.walk(directory) for filename in filenames if filename.endswith(("_img1.png", "_img2.png"))]

    print(f"Total number of samples: {len(image_files)}")
    start_time = time.time()

    for file_path in tqdm(image_files, desc="Adding haze"):
        rel_file_path = os.path.relpath(file_path, directory)

        if rel_file_path in checkpoints and checkpoints[rel_file_path] == True:
            print(f"Skipping already processed image: {file_path}")
        else:
            depth_map = create_depth_map(file_path)
            hazy_image = add_haze_to_image(file_path, depth_map, k, beta)
            hazy_image.save(file_path)
            num_samples += 1
            checkpoints[rel_file_path] = True

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoints, f)

    end_time = time.time()
    print(f"Added haze to {num_samples} FlyingChairsOcc samples in {end_time - start_time:.2f} seconds")

    return num_samples


def add_haze_to_flyingthings3d_subset(directory, checkpoint_file, k=0.5, beta=0.00008):
    num_samples = 0
    checkpoints = {}

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoints = json.load(f)
    image_files = [os.path.join(dirpath, filename) for dirpath, _, filenames in os.walk(directory) for filename in filenames if filename.endswith(".png")]

    print(f"Total number of samples: {len(image_files)}")
    start_time = time.time()

    for file_path in tqdm(image_files, desc="Adding haze"):
        rel_file_path = os.path.relpath(file_path, directory)

        if rel_file_path in checkpoints and checkpoints[rel_file_path] == True:
            print(f"Skipping already processed image: {file_path}")
        else:
            depth_map = create_depth_map(file_path)
            hazy_image = add_haze_to_image(file_path, depth_map, k, beta)
            hazy_image.save(file_path)
            num_samples += 1
            checkpoints[rel_file_path] = True

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoints, f)

    end_time = time.time()
    print(f"Added haze to {num_samples} FlyingThings3D_subset samples in {end_time - start_time:.2f} seconds")

    return num_samples

def is_directory_structure_identical(src, dst):
    for src_root, src_dirs, src_files in os.walk(src):
        dst_root = src_root.replace(src, dst)

        if not os.path.exists(dst_root):
            return False

        dst_dirs, dst_files = [], []
        for _, dirs, files in os.walk(dst_root):
            dst_dirs = dirs
            dst_files = files
            break

        if set(src_dirs) != set(dst_dirs) or set(src_files) != set(dst_files):
            return False

    return True


displayed_samples = False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Add haze to images using depth maps.')
    parser.add_argument('--beta', type=float, default=1, help='Haze scattering coefficient value')
    parser.add_argument('--k', type=float, default=0.5, help='Atmospheric light contribution k')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to display (default: 1)')

    parser.add_argument('--src_dir', type=str, required=True, help='Source directory of the dataset')
    parser.add_argument('--dst_dir', type=str, required=True, help='Destination directory for the hazy dataset')
    parser.add_argument('--dataset', type=str, required=True, choices=['sintel', 'kitti15', 'flyingchairs', 'flyingthings3d', 'hd1k', 'chairssdhom', 'flyingchairsocc'], help='Dataset to be used (sintel or kitti)')

    args = parser.parse_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    dataset = args.dataset
    beta = args.beta
    k = args.k
    num_samples_arg = args.num_samples


    checkpoint_file = f'{dataset}_checkpoints.json'

    if dataset == 'kitti15':
        # Add KITTI-specific code here
        if not displayed_samples:
            print(f"Displaying samples from {src_dir}...")
            display_kitti15_samples(src_dir, num_samples_arg)
            displayed_samples = True

    elif dataset == 'sintel':
        # Add Sintel-specific code here
        if not displayed_samples:
            print(f"Displaying samples from {src_dir}...")
            display_sintel_samples(src_dir, num_samples_arg)
            displayed_samples = True
    elif dataset == 'flyingchairs':
        # Add FlyingChairs-specific code here
        if not displayed_samples:
            print(f"Displaying samples from {src_dir}...")
            display_flyingchairs_samples(src_dir, num_samples_arg)
            displayed_samples = True

    elif dataset == 'flyingthings3d':
        if not displayed_samples:
            print(f"Displaying samples from {src_dir}...")
            display_flyingthings3d_samples(src_dir, num_samples_arg)
            displayed_samples = True

    elif dataset == 'hd1k':
        if not displayed_samples:
            print(f"Displaying samples from {src_dir}...")
            display_hd1k_samples(src_dir, num_samples_arg)
            displayed_samples = True

    elif dataset == 'chairssdhom':
        if not displayed_samples:
            print(f"Displaying samples from {src_dir}...")
            display_chairssdhom_samples(src_dir, num_samples_arg)
            displayed_samples = True

    elif dataset == 'flyingchairsocc':
        if not displayed_samples:
            print(f"Displaying samples from {src_dir}...")
            display_flyingchairssocc_samples(src_dir, num_samples_arg)
            displayed_samples = True

    elif dataset == 'flyingthings3d_subset':
        if not displayed_samples:
            print(f"Displaying samples from {src_dir}...")
            display_flyingthings3d_subset_samples(src_dir, num_samples_arg)
            displayed_samples = True

    proceed = input("Do you want to proceed with cloning and hazing the dataset? (y/n): ")

    if proceed.lower() != 'y':
        print("Aborted.")
        exit()

    if os.path.exists(dst_dir):
        if is_directory_structure_identical(src_dir, dst_dir):
            print(f"Directory structure already exists in {dst_dir}. Skipping cloning.")
        else:
            shutil.rmtree(dst_dir)
            print(f"Removed existing directory: {dst_dir}")
            print(f"Cloning directory {src_dir} to {dst_dir}...")
            clone_directory(src_dir, dst_dir)
            print(f"Clone completed.")
    else:
        print(f"Cloning directory {src_dir} to {dst_dir}...")
        clone_directory(src_dir, dst_dir)
        print(f"Clone completed.")

    print(f"Starting to add haze to images in {dst_dir}...")
    if dataset == 'kitti15':
        num_samples = add_haze_to_kitti_images_in_directory(dst_dir, checkpoint_file, k, beta)
    elif dataset == 'sintel':
        num_samples = add_haze_to_sintel_images_in_directory(dst_dir, checkpoint_file, k, beta)
    # print(f"Adding haze completed: {num_samples} samples processed")

    elif dataset == 'flyingchairs':
        num_samples = add_haze_to_flyingchairs_images_in_directory(dst_dir, checkpoint_file, k, beta)
    
    elif dataset == 'flyingthings3d':
        num_samples = add_haze_to_flyingthings3d_images_in_directory(dst_dir, checkpoint_file, k, beta)
    
    elif dataset == 'hd1k':
        num_samples = add_haze_to_hd1k_images_in_directory(dst_dir, checkpoint_file, k, beta)

    elif dataset == 'chairssdhom':
        num_samples = add_haze_to_chairssdhom_images_in_directory(dst_dir, checkpoint_file, k, beta)

    elif dataset == 'flyingchairsocc':
        num_samples = add_haze_to_flyingchairsocc(dst_dir, checkpoint_file, k, beta)
    
    elif dataset == 'flyingthings3d_subset':
        num_samples = add_haze_to_flyingthings3d_subset(dst_dir, checkpoint_file, k, beta)
    
    print(f"Adding haze completed: {num_samples} samples processed")