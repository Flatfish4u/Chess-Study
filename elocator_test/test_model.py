import torch
import os
from complexity.model import ChessModel

def test_model_load():
    print("Testing model loading...")
    
    # Create model
    model = ChessModel()
    
    # Use absolute path
    model_path = '/Users/benjaminrosales/Desktop/Chess_Study_Coding/elocator_test/complexity/models/model.pth'
    print(f"Attempting to load model from: {model_path}")
    
    try:
        # Load with weights_only=True to address the warning
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        print("✓ Model loaded successfully!")
        
        # Set model to evaluation mode
        model.eval()
        
        # Create a batch of 2 samples instead of 1
        dummy_input = torch.randn(2, 780)  # Two samples of 780 dimensions
        
        with torch.no_grad():  # Disable gradient computation for inference
            output = model(dummy_input)
            
        print("✓ Model forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output values: {output.squeeze().tolist()}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    test_model_load()