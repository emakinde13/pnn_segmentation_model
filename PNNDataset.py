from pathlib import Path
import skimage as ski
from torch.utils.data import Dataset
import cv2

class PNNDataset(Dataset):
    def __init__(self, candidate_dir : str, mask_dir : str, transforms = None):
        
        # Create a list of the path objects for each og image
        self.candidate_dir = Path(candidate_dir)
        self.candidate_paths = list(self.candidate_dir.glob("*.png"))

        # Create a dict which will later help us couple og images with their respective masks
        self.mask_dir = Path(mask_dir)
        
        self.mask_dict = {
            p.name: p
            for p in Path(self.mask_dir).glob("*.png")
        }
        
        self.transforms = transforms

    def __len__(self):
        return len(self.candidate_paths)
    
    def __getitem__(self, idx):
        imagePath = self.candidate_paths[idx]
        image = cv2.imread(str(imagePath), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_dict[imagePath.name]), cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        
        return (image, mask)

        


