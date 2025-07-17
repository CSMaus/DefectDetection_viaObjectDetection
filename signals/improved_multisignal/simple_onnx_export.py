from detection_models.complex_detection_model import ComplexDetectionModel
from detection_models.complex_onnx import ComplexDetectionModelONNX
from detection_models.complex_fix import ComplexDetectionModelFix
from detection_models.noise_robust_tr2 import NoiseRobustDetectionModel
from detection_models.pattern_embedding import PatternEmbeddingModel
from detection_models.enhanced_pattern import EnhancedPatternModel
from detection_models.direct_defect import DirectDefectModel



import torch

def export_original_model_to_onnx(model_path, onnx_model_path, signal_length=320):
    """Export original model without modifications - accept ONNX limitations"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load original model exactly as trained
    # model = ComplexDetectionModel(signal_length=signal_length).to(device)
    # model = ComplexDetectionModelFix(signal_length=signal_length).to(device)
    # model = ComplexDetectionModelFix(signal_length=signal_length).to(device)
    # model = NoiseRobustDetectionModel(signal_length=signal_length).to(device)
    # model = PatternEmbeddingModel(signal_length=signal_length).to(device)
    # model = EnhancedPatternModel(signal_length=signal_length).to(device)
    model = DirectDefectModel(signal_length=signal_length,
                                               d_model=64,
                                               num_heads=16,
                                               num_layers=6,
                                               dropout=1).to(device)

    # d_model=128, num_heads=8, num_layers=6, dropout=0.5  - this is for 008
    # 006 is the first simplest one with d_model=64, num_heads=4, num_layers=4, dropout=0.1
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'Unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint directly")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 50, signal_length).to(device)
    
    # Test with original model
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Original model output shape: {output.shape}")
        print(f"Original model output range: {output.min().item():.4f} to {output.max().item():.4f}")
    
    # Try different ONNX export settings to find one that works
    export_configs = [
        {"opset": 11, "dynamic": False},
        {"opset": 13, "dynamic": False}, 
        {"opset": 15, "dynamic": False},
        {"opset": 11, "dynamic": True},
    ]
    
    for i, config in enumerate(export_configs):
        try:
            print(f"\nTrying export config {i+1}: opset={config['opset']}, dynamic_axes={config['dynamic']}")
            
            export_args = {
                "model": model,
                "args": dummy_input,
                "f": f"{onnx_model_path[:-5]}_v{i+1}.onnx",
                "export_params": True,
                "opset_version": config["opset"],
                "input_names": ['input'],
                "output_names": ['detection_prob']
            }
            
            if config["dynamic"]:
                export_args["dynamic_axes"] = {
                    'input': {0: 'batch_size'},
                    'detection_prob': {0: 'batch_size'}
                }
            
            torch.onnx.export(**export_args)
            print(f"✅ SUCCESS: Exported to {export_args['f']}")
            return export_args['f']
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)[:100]}...")
            continue
    
    print("❌ All export attempts failed!")
    return None

# Main execution
# modelname = "ComplexDetectionModel"
modelname = "DirectDefectModel"
attempt = "010"
# model_path = f'models/ComplexONNX_20250717_1746/best_complexonnx_detection.pth'
# model_path = f'models/NoiseRobust_20250717_1838/best_noiserobust_detection.pth'
# model_path = f'models/DirectDefectModel_20250717_2155/best_detection.pth'
model_path = f'models/DirectDefectModel_20250718_0001/best_detection.pth'

successful_export = export_original_model_to_onnx(
    model_path, 
    f'models/{attempt}-{modelname}.onnx'
)

if successful_export:
    print(f"\n🎉 Use this ONNX file: {successful_export}")
else:
    print("\n💥 ONNX export failed completely - the model architecture is not ONNX compatible")
