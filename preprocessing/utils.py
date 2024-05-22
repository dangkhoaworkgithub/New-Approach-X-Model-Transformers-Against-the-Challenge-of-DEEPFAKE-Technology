import os
import shutil
from PIL import Image
import random

def format_folders(input_paths, output_path, subfolder_name):
    for folder in ['train', 'test', 'val']:
        os.makedirs(os.path.join(output_path, folder, subfolder_name), exist_ok=True)

    all_photos = []
    for folder in input_paths:
        for file in os.listdir(folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                all_photos.append(os.path.join(folder, file))

    random.shuffle(all_photos)
    total_photos = len(all_photos)
    train_split = int(total_photos * 0.7)
    test_split = int(total_photos * 0.15)

    train_photos = all_photos[:train_split]
    test_photos = all_photos[train_split:train_split + test_split]
    val_photos = all_photos[train_split + test_split:]

    def process_and_move(photos, folder):
        count = 0
        for photo in photos:
            try:
                with Image.open(photo) as img:
                    if img.format != 'JPEG':
                        photo = os.path.splitext(photo)[0] + '.jpg'
                        img = img.convert('RGB')

                    if img.size != (224, 224):
                        img = img.resize((224, 224), Image.Resampling.LANCZOS)

                    save_path = os.path.join(output_path, folder, subfolder_name, os.path.basename(photo))
                    img.save(save_path)
                    count += 1
            except Exception as e:
                print(f"Error processing {photo}: {e}")

        return count

    moved_train = process_and_move(train_photos, 'train')
    moved_test = process_and_move(test_photos, 'test')
    moved_val = process_and_move(val_photos, 'val')

    return moved_train, moved_test, moved_val

def images_copy(source_directory, destination_directory, max_images=80000):
    os.makedirs(destination_directory, exist_ok=True)

    image_count = 0

    for root, dirs, files in os.walk(source_directory):
        for file in files:
            if file.lower().endswith('.jpg'):
                if image_count < max_images:
                    source_file_path = os.path.join(root, file)
                    destination_file_path = os.path.join(destination_directory, file)

                    shutil.copy2(source_file_path, destination_file_path)
                    print(f"Copied {source_file_path} to {destination_file_path}")

                    image_count += 1
                else:
                    print("Reached the limit images to copy. Exiting...")
                    return
