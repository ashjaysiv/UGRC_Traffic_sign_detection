import cv2
import numpy as np
import os


def add_random_blur(image, max_kernel_size=25):
    """Add random blur to an image."""
    kernel_size = np.random.randint(1, max_kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure an odd kernel size for GaussianBlur
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

def add_sunlight_patches(image, num_patches=10, patch_size=(100, 100), intensity_range=(0.1, 0.3)):
    """Add patches of sunlight exposure to an image."""
    image_with_sunlight = image.copy()

    for _ in range(num_patches):
        # Randomly select a position for the sunlight patch
        x = np.random.randint(0, image.shape[1] - patch_size[1])
        y = np.random.randint(0, image.shape[0] - patch_size[0])

        # Create a sunlight patch with random intensity
        sunlight_patch = np.ones(patch_size + (image.shape[2],), dtype=np.float64)
        intensity = np.random.uniform(*intensity_range)
        sunlight_patch *= intensity

        # Blend the sunlight patch with the original image
        image_with_sunlight[y:y+patch_size[0], x:x+patch_size[1]] = cv2.addWeighted(
            image[y:y+patch_size[0], x:x+patch_size[1]],
            1 - intensity, sunlight_patch, intensity, 0, dtype=cv2.CV_8U 
        )

    return image_with_sunlight

def augment_image(image):
    """Apply random transformations to augment the input image."""
    counter = 0
    for angle in range(-60,60, 15):

        rows, cols, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))


        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            scaled_image = cv2.resize(rotated_image, None, fx=scale, fy=scale)

            white_canvas = np.ones_like(scaled_image) * 255  

            for brightness_factor in [0.1, 0.2, 0.4, 0.7, 0.8]:
                hsv_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv_image)
                v_bright = np.clip(v * brightness_factor, 0, 255).astype(np.uint8)  # Increase as needed for extreme brightness
                hsv_bright = cv2.merge([h, s, v_bright])
                brightened_image = cv2.cvtColor(hsv_bright, cv2.COLOR_HSV2BGR)
                brightened_image = add_random_blur(brightened_image)
                # brightened_image = add_sunlight_patches(brightened_image)
                path = os.path.join(r"C:\Users\ashmi\OneDrive\traffic_sign_detection\main\augment_data", f"{counter}.jpg")
                # path = r"C:\Users\ashmi\OneDrive\\main\augment_data\"+ str(counter) + ".jpg"
                cv2.imwrite(path, brightened_image)
                counter += 1

original_image = cv2.imread(r"C:\Users\ashmi\OneDrive\traffic_sign_detection\zip\VGG\GTSRB\train\Intersection\test_vdo_2.mkv_frames_7_crop.jpg")

# cv2.imshow("hh", original_image)
# cv2.waitKey(0)
augment_image(original_image)


# # brightness
# for scale_factor in np.arange(0,1, 0.1):
#     image = original_image
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv_image)
#     v_bright = np.clip(v * scale_factor, 0, 255).astype(np.uint8)  # Increase as needed for extreme brightness
#     hsv_bright = cv2.merge([h, s, v_bright])
#     bright_image = cv2.cvtColor(hsv_bright, cv2.COLOR_HSV2BGR)

#     path = os.path.join(r"C:\Users\ashmi\OneDrive\traffic_sign_detection\main\augment_data", f"{scale_factor}.jpg")
#     cv2.imwrite(path, bright_image)


# # whiteness
# white_canvas = np.ones_like(original_image) * 255  
# for alpha in [0, 0.9]:

#     alpha = 0.9 
#     white_image = cv2.addWeighted(original_image, 1 - alpha, white_canvas, alpha, 0)



