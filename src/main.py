import cv2 as cv
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from loguru import logger

from utils.utils import fetch_input_dir, found_images_list


class YoloAnalysis():
    '''
    Use Yolo v11 to detect targets in images
    Args: list of path of images and model
    ''' 
    def __init__(self, images_paths: list[Path], model):
        self.imgs_paths = images_paths
        self.model = model

    def prediction(
        self, img_path: Path,
        id_class: int, conf: float = 0.7, 
        show: bool = False
    ) -> bool:
        '''
        Open image then detect people using Yolo functions
        '''
        img = cv.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        # Take only bbox for person
        result = self.model(
            img, conf=conf, classes=[id_class], verbose=False
        )
        for r in result:
            if r.boxes is not None and len(r.boxes) > 0:
                if show:
                    logger.info(str(img_path))
                    img = r.plot()
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    plt.imshow(img)
                    plt.axis("off")
                    plt.show()
                    plt.close()
                return True

        return False
    

    def run_analysis(self):
        '''
        Core of analysis: apply Yolo to a list of image
        if img has person, put this one in that list
        '''
        
        # Define number of "target" class 
        person_class_id = next(
            k for k, v in self.model.names.items() if v == "person"
        )
        logger.info('Using model to detect target')
        target_list = []
        for img_path in self.imgs_paths:
            has_target = self.prediction(
                img_path, person_class_id, show=True,
            )
            if has_target:
                target_list.append(img_path)
        
        return target_list



def main():
    selected_dir = fetch_input_dir()
    if selected_dir:
        filelist = found_images_list(selected_dir)
        # Find the folder in which is contained the model
        ROOT = Path(__file__).resolve().parents[1]
        MODEL_PATH = ROOT / "training_models" / "yolo11x.pt"

        model = YOLO(MODEL_PATH)
        obj = YoloAnalysis(filelist, model)
        imgs_list = obj.run_analysis()        


if __name__ == "__main__":
    main()