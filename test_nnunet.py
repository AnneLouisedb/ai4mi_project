import torch
from nnUnet.nnUnet import nnUNet

def test_forward():
    model = nnUNet(in_channels=1, out_channels=4)
    model.init_weights()
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 128, 128, 128)  # Adjust dimensions as per your data
        output = model(dummy_input)
        assert output.shape == (1, 4, 128, 128, 128), f"Unexpected output shape: {output.shape}"
    print("nnUNet forward pass successful.")

if __name__ == "__main__":
    test_forward()
