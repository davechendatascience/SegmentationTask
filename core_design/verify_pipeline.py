import torch
import torch.optim as optim
from core_design.model_block import CLIPPanopticModel
from core_design.loss_block import SetCriterion, HungarianMatcher

# Configuration
IMAGE_SIZE = 224 # Smaller for quick test
BATCH_SIZE = 2
NUM_CLASSES = 150
DEVICE = "cpu" # Test on CPU for simplicity in this environment

def test_pipeline():
    print("Initializing Model...")
    model = CLIPPanopticModel(num_classes=NUM_CLASSES)
    model.to(DEVICE)
    
    print("Initializing Loss...")
    matcher = HungarianMatcher()
    criterion = SetCriterion(NUM_CLASSES, matcher, {'loss_ce': 1.0, 'loss_mask': 1.0, 'loss_dice': 1.0}).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    print("Generating Synthetic Data...")
    # Random Images: [B, 3, H, W]
    pixel_values = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    
    # Random Targets
    targets = []
    for i in range(BATCH_SIZE):
        # 2 objects per image
        masks = torch.randint(0, 2, (2, IMAGE_SIZE, IMAGE_SIZE)).float().to(DEVICE)
        labels = torch.randint(0, NUM_CLASSES, (2,)).to(DEVICE)
        targets.append({"masks": masks, "class_labels": labels})
    
    print("Running Forward Pass...")
    outputs = model(pixel_values)
    
    print("Calculating Loss...")
    loss, loss_dict = criterion(outputs, targets)
    
    print(f"Loss: {loss.item()}")
    
    print("Running Backward Pass...")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Integration Test Passed!")

if __name__ == "__main__":
    test_pipeline()
