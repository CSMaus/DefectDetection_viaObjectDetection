import torch
import numpy as np
import torch.nn as nn
import os


class DefectDetectionModel(nn.Module):
    def __init__(self, signal_length, num_signals_per_set):
        super(DefectDetectionModel, self).__init__()

        # Feature extractor for single-channel signals
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1, stride=1),
            nn.ReLU()
        )

        # Transformer encoder for sequence of signals
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True),
            num_layers=4
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, signal_length, num_signals = x.size()

        x = x.permute(0, 2, 1).contiguous().view(-1, 1, signal_length)  # [batch_size * num_signals, 1, signal_length]

        x = self.feature_extractor(x)  # [batch_size * num_signals, 128, reduced_length]
        x = x.mean(dim=2)  # Global average pooling over signal length: [batch_size * num_signals, 128]
        x = x.view(batch_size, num_signals, -1)  # [batch_size, num_signals, 128]
        x = self.transformer_encoder(x)  # [batch_size, num_signals, 128]
        x = self.classifier(x).squeeze(-1)  # [batch_size, num_signals]

        return x


def export_model_to_onnx_with_dynamic_signals_WRONG(model, model_path, onnx_model_path, signal_length, hidden_sizes):
    # Load the pre-trained model weights
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode (important for ONNX export)
    model.eval()

    # Create a dummy input with batch size of 1 and a dynamic number of signals
    dummy_input = torch.randn(1, signal_length, num_signals_per_set)  # 300 is arbitrary;
    # ONNX will allow any dynamic value for this dimension

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


def export_model_to_onnx_with_dynamic_signals(model, model_path, onnx_model_path, signal_length):
    # Load the pre-trained model weights
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input with batch size of 1 and a dynamic number of signals
    num_signals_per_set = 300  # Example dynamic number of signals
    dummy_input = torch.randn(1, signal_length, num_signals_per_set)

    # Export the model to ONNX with dynamic axes
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'num_signals_per_set'},  # Dynamic batch and number of signals
            'output': {0: 'batch_size', 1: 'num_signals_per_set'}  # Match output dimensions
        }
    )


signal_length = 320
num_signals_per_set = 300

model = DefectDetectionModel(signal_length, num_signals_per_set)
modelname = "Conv1d_OPD"
export_model_to_onnx_with_dynamic_signals(model, f'models/MSC_model{modelname}.pth',
                                          f'models/MSC_model{modelname}.onnx', signal_length)
                                          # 'MultiSignalClassifier4_dynamic.onnx', signal_length, hidden_sizes)


