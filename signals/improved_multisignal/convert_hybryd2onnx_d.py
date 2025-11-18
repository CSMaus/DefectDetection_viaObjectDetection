from detection_models.hybrid_binary_dynamic import HybridBinaryModel
import torch
import os


def load_legacy_mha_checkpoint_into_tiny(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    old_sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    new_sd = model.state_dict()

    # 1) copy everything except self_attn.*
    for k in new_sd.keys():
        if ".self_attn." in k:
            continue
        if k in old_sd:
            new_sd[k] = old_sd[k]

    # 2) decompose in_proj/out_proj into q,k,v,o for each layer
    num_layers = len(model.transformer_layers)
    for i in range(num_layers):
        prefix_old = f"transformer_layers.{i}.self_attn"
        prefix_q   = f"transformer_layers.{i}.self_attn.q"
        prefix_k   = f"transformer_layers.{i}.self_attn.k"
        prefix_v   = f"transformer_layers.{i}.self_attn.v"
        prefix_o   = f"transformer_layers.{i}.self_attn.o"

        Wqkv = old_sd[prefix_old + ".in_proj_weight"]   # [3D, D]
        bqkv = old_sd[prefix_old + ".in_proj_bias"]     # [3D]
        Wout = old_sd[prefix_old + ".out_proj.weight"]  # [D, D]
        bout = old_sd[prefix_old + ".out_proj.bias"]    # [D]

        D = Wout.shape[0]

        Wq = Wqkv[0:D, :]
        Wk = Wqkv[D:2*D, :]
        Wv = Wqkv[2*D:3*D, :]
        bq = bqkv[0:D]
        bk = bqkv[D:2*D]
        bv = bqkv[2*D:3*D]

        new_sd[prefix_q + ".weight"] = Wq.clone()
        new_sd[prefix_q + ".bias"]   = bq.clone()
        new_sd[prefix_k + ".weight"] = Wk.clone()
        new_sd[prefix_k + ".bias"]   = bk.clone()
        new_sd[prefix_v + ".weight"] = Wv.clone()
        new_sd[prefix_v + ".bias"]   = bv.clone()
        new_sd[prefix_o + ".weight"] = Wout.clone()
        new_sd[prefix_o + ".bias"]   = bout.clone()

    model.load_state_dict(new_sd, strict=True)
    return model



def export_model_to_onnx(model, device, model_path, onnx_model_path, signal_length):
    model = load_legacy_mha_checkpoint_into_tiny(model, model_path, device)
    model.eval()

    dummy_input = torch.randn(1, 50, signal_length, device=device)  # traced with N=50

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=15,
        do_constant_folding=False,
        input_names=['input'],
        output_names=['defect_prob'],
        dynamic_axes={
            'input':      {0: 'batch_size', 1: 'num_signals'},
            'defect_prob': {0: 'batch_size', 1: 'num_signals'},
        }
    )
    print(f"Model exported to {onnx_model_path}")


if __name__ == "__main__":
    signal_length = 320
    num_heads = 8
    num_transformer_layers = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HybridBinaryModel(
        signal_length=signal_length,
        num_heads=num_heads,
        num_transformer_layers=num_transformer_layers
    ).to(device)

    modelname = "HybridBinaryModel"
    attempt = "013d"
    model_path = 'models/HybridBinaryModel_20251112_1902/best_detection.pth'
    onnx_path = f"models/{attempt}-{modelname}.onnx"

    export_model_to_onnx(model, device, model_path, onnx_path, signal_length)
