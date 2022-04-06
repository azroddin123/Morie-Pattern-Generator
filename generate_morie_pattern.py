import albumentations as A
import cv2
import os
import argparse
import glob
from pathlib import Path
import random
from random import choice
from nb_utils.file_dir_handling import list_files
sys_random = random.SystemRandom()
parser = argparse.ArgumentParser(
    "pass the path of morie image and original images and argparser"
)

parser.add_argument(
    "--original",
    help="orignal image folder path pass here",
    default="/home/azhar/Python/Morie-Pattern-Generator/original-image",
)
parser.add_argument(
    "--morie", default="/home/azhar/Python/Morie-Pattern-Generator/mori"
)
parser.add_argument(
    "--output",
    help="generated image folder path passed here",
    default="/home/azhar/Python/Morie-Pattern-Generator/output_result",
)

args = parser.parse_args()

original_path = args.original
morie_path = args.morie
output = args.output


trans = [0.5,0.6,0.7,0.8]
alpha_val = [0.5,0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85]

if not os.path.isdir(output):
    os.makedirs(output)
    print("created")
if not os.path.isdir(original_path) or not os.path.isdir(morie_path):
    print("Please provide correct path for image folder")

original_images = list_files(args.original, filter_ext=[".png", ".jpg", ".jpeg"])
# print(len(original_images))
morie_images = list_files(morie_path, filter_ext=[".png", ".jpg", ".jpeg"])
# print(len(morie_images))

# Declare an augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=random.choice(trans)),
    A.ShiftScaleRotate(p=random.choice(trans),interpolation = 2),
    A.RandomBrightnessContrast(p=random.choice(trans)),   
])

base_transform = A.Compose ([
    A.MotionBlur(p=0.5),
    A.MedianBlur(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.GaussianBlur(p=0.5),
    A.MotionBlur(p=0.5),
    A.MultiplicativeNoise(0.5),

])

def morie_pattern_generator(base_image):
    i = 1
    b_image = cv2.imread(base_image)
    # b_image = cv2.cvtColor(b_image, cv2.COLOR_BGR2RGB)
    random.seed(10)
    b_image =base_transform(image=b_image)['image']
    p = Path(base_image)
    # alpha = random.choice(alpha_val)
    # print(alpha)
    for mask_image in morie_images:
        alpha = sys_random.choice(alpha_val)
        print(alpha)
        # print(random.shuffle(alpha))
        mask = cv2.imread(mask_image)
        image = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        random.seed(47)
        augmented_image = transform(image=image)['image']
        augmented_image = cv2.resize(augmented_image, (b_image.shape[1],b_image.shape[0]))        
        # cv2.imwrite(f"{output}/{p.stem}{i}{i}.jpg",augmented_image)
        # Generating random alpha value for various effect 
        
        # print(alpha)
        beta = 1.00 - alpha
        gamma = 0.0
        # # masked_img = cv2.addWeighted(image, alpha, mask, beta, gamma)
        augmented_image = cv2.addWeighted(b_image, alpha, augmented_image, beta, gamma)
        # # cv2.imwrite("masked_img.png",masked_img)
        # # cv2.imwrite(f"{output}/{base_image}.png",masked_img)
        cv2.imwrite(f"{output}/{p.stem}{str(alpha)}{i}.jpg", augmented_image)
        i += 1
       
     
        
for image in original_images:
    morie_pattern_generator(image)
    