import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2 as cv
from pathlib import Path
from loguru import logger

import torch
from torchvision import transforms
from PIL import Image

class YoloAnalysis():
    '''
    Use Yolo v11 to detect targets in images
    Args: list of path of images and model path
    ''' 
    def __init__(self, images_paths: list[Path], model:Path):
        self.imgs_paths = images_paths
        self.model = YOLO(model)

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
                img_path, person_class_id, show=False,
            )
            if has_target:
                target_list.append(img_path)
        logger.success('Classification finished')
        return target_list

class ResNet50:
    def __init__(self, topk: int=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.topk = topk
        self.class_names = None
        modelpath = Path.cwd() / "training_models" / "best_model.pt"
        self.model = torch.load(
            modelpath,
            map_location=self.device,
            weights_only=False
        )
        self.model.to(self.device) 
        self.model.eval()

        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        '''
        Core analysis: detect class in the image.
        If self.class_names is not used return
        class number from original train folder
        '''
        
        img = Image.open(image_path).convert("RGB")
        x = self.test_transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)
            probs = torch.softmax(output, dim=1)
            top_probs, top_idxs = probs.topk(self.topk, dim=1)

        top_idxs = top_idxs.cpu().tolist()[0]

        if self.class_names is not None:
            top_classes = []
            for i in top_idxs:
                top_classes.append(self.class_names[i])
            return top_classes
        else:
            # If not defined use only the value list
            return top_idxs 

if __name__ == "__main__":
    img_path = r"C:\Users\Andre-343H\Desktop\test\IMG.jpg"
    class_names = ['A', 'B', 'C', 'D', 'E', 'F']
    obj = ResNet50(topk=3)
    obj.class_names = class_names
    logger.info(obj.predict(img_path))