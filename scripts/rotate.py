from PIL import Image

# Specify the filename
filename = "0.png"

# Open the image
image_path = filename
original_image = Image.open(image_path)

# Define the new width (adjust as needed)
new_width = 500

# Calculate the corresponding height to maintain the original aspect ratio
aspect_ratio = original_image.width / original_image.height
new_height = int(new_width / aspect_ratio)

# Resize the image, keeping the original height
shrunken_image = original_image.resize((new_width, original_image.height))

# Display the original and shrunken images
original_image.show(title='Original Image')
shrunken_image.show(title='Shrunken Image')

# Save the shrunken image with a new name, if ne
