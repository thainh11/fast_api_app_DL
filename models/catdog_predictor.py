import sys
import torch
import torchvision

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image
from torch.nn import functional as F
from utils.logger import Logger
from config.catdog_cfg import CatDogDataConfig
from models.catdog_model import CatsVsDogsModels

LOGGER = Logger(__file__, log_file = "predictor.log")
LOGGER.log.info("Starting predictor")

class Predictor:
    def __init__(
        self, 
        model_name: str,
        model_path: str,
        device: str = "cuda"):
        
        self.model = CatsVsDogsModels(CatDogDataConfig.N_CLASSES)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.model_name = model_name
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.create_transforms()
    
    async def predict(self, image):
        pil_img = Image.open(image)
        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')
        transformed_image = self.transforms_(pil_img).unsqueeze(0)
        output = await self.model_inference(transformed_image)
        probs, best_prob, predicted_id, predicted_class = self.output2pred(output)
        LOGGER.log_model(self.model_name)
        LOGGER.log_response(best_prob, predicted_id, predicted_class)
        
        torch.cuda.empty_cache()
        
        resp_dict = {
            'probs': probs,
            'best_prob': best_prob,
            'predicted_id': predicted_id,
            'predicted_class': predicted_class,
            'predictor_name': self.model_name
        }
        
        return resp_dict
    async def model_inference(self, input):
        input = input.to(self.device)
        # with torch.no_grad():
        output = self.model(input)
        return output
    
    def load_model(self):
        try:
            model = CatsVsDogsModels(CatDogDataConfig.N_CLASSES)
            model.load_state_dict(torch.load(self.model_weight, map_location=self.device))
            model.to(self.device)
            model.eval()
        
        except Exception as e:
            LOGGER.log.error(f"Load model failed")
            LOGGER.log.error(f"Error: {e}")
            
            return None
    
    def create_transforms(self):
        self.transforms_ = torchvision.transforms.Compose([
            torchvision.transforms.Resize(CatDogDataConfig.IMG_SIZE),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=CatDogDataConfig.NORMALIZE_MEAN, 
                std=CatDogDataConfig.NORMALIZE_STD
                )
        ])
    
    def output2pred(self, output):
        probabilities = F.softmax(output, dim=1)
        best_prob = torch.max(probabilities, 1)[0].item()
        predicted_id = torch.max(probabilities, 1)[1].item()
        predicted_class = CatDogDataConfig.ID2LABEL[predicted_id]
        return probabilities.squeeze().tolist(), round(best_prob, 6), predicted_id, predicted_class