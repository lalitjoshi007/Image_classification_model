import torch
from transformers import  BlipProcessor, BlipForConditionalGeneration

# Set device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

class ModelLoader:
    """Loads and caches models for video processing."""
    
    def __init__(self):
        # BLIP Model
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    def get_models(self):
        """Returns the loaded models."""
        return  self.blip_model, self.blip_processor

# Global instance for efficient reuse
model_loader = ModelLoader()
