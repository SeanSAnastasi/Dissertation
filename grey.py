import os
from PIL import Image
from tqdm import tqdm

def grayscale_images(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        if "grayscale" in dirs:
            # If the current directory contains a "grayscale" folder, skip it and its subdirectories
            dirs.remove("grayscale")
            continue
        for file in tqdm(files, desc="Processing images in " + subdir):
            filepath = os.path.join(subdir, file)
            if filepath.endswith(".jpg") or filepath.endswith(".jpeg") or filepath.endswith(".png"):
                # Load the image and convert it to grayscale
                img = Image.open(filepath).convert("L")

                # Create a new directory for the grayscale images
                grayscale_dir = os.path.join(os.path.dirname(filepath), "grayscale")
                os.makedirs(grayscale_dir, exist_ok=True)

                # Save the grayscale image in the new directory
                grayscale_filepath = os.path.join(grayscale_dir, file)
                img.save(grayscale_filepath)

    print("Grayscale conversion complete.")

grayscale_images('clusters/')
grayscale_images('method2/')
