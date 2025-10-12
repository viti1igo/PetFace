import os
import random
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class CatPairDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, 
                 subset_size=5000, pairs_per_epoch=20000):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.subset_size = subset_size
        self.pairs_per_epoch = pairs_per_epoch
        
        # Group images by cat ID
        self.cat_groups = defaultdict(list)
        for i, label in enumerate(labels):
            self.cat_groups[label].append(i)
        
        self.cat_ids = list(self.cat_groups.keys())
        print(f'Total cats: {len(self.cat_ids)}')
        print(f'Total images: {len(image_paths)}')
        
        # Generate pairs
        self._generate_pairs()
    
    def _generate_pairs(self):
        """
        Generate positive and negative pairs
        """
        self.pairs = []
        self.pair_labels = []
        
        # Sample subset of cats
        selected_cats = random.sample(self.cat_ids, 
                                    min(self.subset_size, len(self.cat_ids)))
        
        positive_pairs = []
        
        # Generate positive pairs (same cat)
        for cat_id in selected_cats:
            images = self.cat_groups[cat_id]
            if len(images) >= 2:
                for i in range(len(images)):
                    for j in range(i+1, min(i+4, len(images))):
                        positive_pairs.append((images[i], images[j], 1))
        
        # Sample if too many
        if len(positive_pairs) > self.pairs_per_epoch // 2:
            positive_pairs = random.sample(positive_pairs, self.pairs_per_epoch // 2)
        
        # Generate negative pairs (different cats)
        negative_pairs = []
        for _ in range(len(positive_pairs)):
            cat1, cat2 = random.sample(selected_cats, 2)
            img1 = random.choice(self.cat_groups[cat1])
            img2 = random.choice(self.cat_groups[cat2])
            negative_pairs.append((img1, img2, 0))
        
        # Combine
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        self.pairs = [(p[0], p[1]) for p in all_pairs]
        self.pair_labels = [p[2] for p in all_pairs]
        
        print(f'Generated {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs')
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        label = self.pair_labels[idx]
        
        # Load images
        img1 = Image.open(self.image_paths[idx1]).convert('RGB')
        img2 = Image.open(self.image_paths[idx2]).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)
    
def load_cat_data(base_path='./cat'):
    """
    Load cat images from PetFace dataset
    Structure: cat/000000/, cat/000001/, ..., cat/164099/
    """
    print(f'Loading data from: {base_path}')
    
    image_paths = []
    labels = []
    
    # Get all cat folders (000000 to 164099)
    cat_folders = sorted([d for d in os.listdir(base_path) 
                         if os.path.isdir(os.path.join(base_path, d))])
    
    print(f'Found {len(cat_folders)} cat folders')
    
    for cat_folder in tqdm(cat_folders, desc='Loading cats'):
        cat_id = int(cat_folder)  # Convert "000000" → 0
        folder_path = os.path.join(base_path, cat_folder)
        
        # Get all images in this cat's folder
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img in images:
            image_paths.append(os.path.join(folder_path, img))
            labels.append(cat_id)
    
    print(f'\nLoaded {len(image_paths)} images from {len(set(labels))} unique cats')
    return image_paths, labels

# if __name__ == '__main__':
#     image_paths, labels = load_cat_data('./cat')