import torch
import numpy as np
import torch.nn as nn
import os


class MultiSignalClassifier(nn.Module):
    def __init__(self, signal_length, hidden_sizes):
        super(MultiSignalClassifier, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(signal_length, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(hidden_sizes[1], num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()
        x = x.view(batch_size * num_signals, signal_length)
        shared_out = self.shared_layer(x)
        shared_out = shared_out.view(batch_size, num_signals, -1)

        attn_output, _ = self.attention(shared_out, shared_out, shared_out)

        outputs = self.classifier(attn_output)
        return outputs.squeeze(-1)


def export_model_to_onnx_with_dynamic_signals(model, model_path, onnx_model_path, signal_length, hidden_sizes):
    # Load the pre-trained model weights
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode (important for ONNX export)
    model.eval()

    # Create a dummy input with batch size of 1 and a dynamic number of signals
    dummy_input = torch.randn(1, 298,
                              signal_length)  # 298 is arbitrary; ONNX will allow any dynamic value for this dimension

    # Export the model to ONNX with dynamic axes for the number of signals
    torch.onnx.export(
        model,
        dummy_input,  # Dummy input
        onnx_model_path,  # Where to save the ONNX model
        export_params=True,  # Store the trained parameters
        opset_version=12,  # ONNX version
        input_names=['input'],  # Input name
        output_names=['output'],  # Output name
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'num_signals'},  # Dynamic batch size and dynamic number of signals
            'output': {0: 'batch_size', 1: 'num_signals'}
        }
    )


signal_length = 360  # 360  # 320
hidden_sizes = [128, 64, 32]
modelname = "FPD"  # "FPD"

model = MultiSignalClassifier(signal_length, hidden_sizes)
export_model_to_onnx_with_dynamic_signals(model, f'MultiSignalClassifier_model{modelname}.pth',
                                          f'test-{modelname}.onnx', signal_length, hidden_sizes)

# export_model_to_onnx_with_dynamic_signals(model, f'MultiSignalClassifier_model{modelname}.pth',
#                                           f'MultiSignalClassifier4_model{modelname}.onnx', signal_length, hidden_sizes)
                                          # 'MultiSignalClassifier4_dynamic.onnx', signal_length, hidden_sizes)


