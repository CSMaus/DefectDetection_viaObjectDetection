from NN_models import MultiSignalClassifier, MultiSignalClassifier_N
import torch

def export_model_to_onnx(model, model_path, onnx_model_path, signal_length, hidden_sizes, num_heads=4):
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dummy_input = torch.randn(1, 298, signal_length)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=12,
        input_names=['input'],
        output_names=['defect_prob', 'defect_start', 'defect_end'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'num_signals'},
            'defect_prob': {0: 'batch_size', 1: 'num_signals'},
            'defect_start': {0: 'batch_size', 1: 'num_signals'},
            'defect_end': {0: 'batch_size', 1: 'num_signals'},
        }
    )


signal_length = 320  # 360  # 320
hidden_sizes = [128, 64, 32]
num_heads = 8  # 4
modelname = "OPD"  # "FPD"
attempt = "008"

# model = MultiSignalClassifier(signal_length, hidden_sizes)
model = MultiSignalClassifier_N(signal_length, hidden_sizes, num_heads)
export_model_to_onnx(model, f'models/{attempt}-_N-MultiSignalClassifier_model{modelname}.pth',
                     f'models/{attempt}-_N-MultiSignalClassifier_model{modelname}.onnx', signal_length, hidden_sizes)
