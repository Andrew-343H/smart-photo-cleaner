from loguru import logger

from utils.utils import fetch_user_dir, found_images_list, add_more, move_to_folder, extract_config
from modules.ai_model import ResNet50

def main():
    filelist = []
    first_folder = None
    selected_dirs = set()
    imgs_list = []
    
    # Extract from config the info
    logger.debug('Find config file..')
    config = extract_config('config_phcleaner.yaml')
    class_target = set(config['cnn']['target_classes'])
    class_names = config['cnn']['classes']
    topk = config['cnn']['top_k']

    # Section dedicated to find all images path
    while True:
        input_dir = fetch_user_dir("Select main folder")
        if not input_dir:
            break
        if first_folder is None:
            first_folder = input_dir
        # Canonical path
        input_dir = input_dir.resolve()
        if input_dir in selected_dirs:
            logger.warning("Folder already selected, skipping.")
            continue

        selected_dirs.add(input_dir)
        filelist.extend(found_images_list(input_dir))
        
        if not add_more():
            break

    # Section dedicated to classify images
    obj = ResNet50(topk=topk)
    obj.class_names = class_names
    
    for img_path in filelist:
        has_target = False
        value = obj.predict(img_path)
        for v in value:
            if v in class_target:
                has_target = True
                break

        if has_target:
            imgs_list.append(img_path)
    logger.success('Classification finished')

    # Section dedicated to move images
    out_title = "Select where the photo will be stored"
    while True:
        out_dir = fetch_user_dir(out_title)
        if out_dir is not None:
            break
        logger.error("User didn't choose a folder, retry")
    
    move_to_folder(first_folder, out_dir, imgs_list)


if __name__ == "__main__":
    main()