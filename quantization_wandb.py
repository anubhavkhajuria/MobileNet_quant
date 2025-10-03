import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torchvision
import torchvision.transforms as transforms
import wandb
import os
import argparse
import math



#model is hapan

class BaseBlock(nn.Module):
    def __init__(self, input_channel, output_channel, alpha, t=6, downsample=False):
        super(BaseBlock, self).__init__()
        self.stride = 2 if downsample else 1
        self.shortcut = (not downsample) and (input_channel == output_channel)
        effective_input_channel = int(alpha * input_channel)
        effective_output_channel = int(alpha * output_channel)
        expansion_channel = t * effective_input_channel
        self.layers = nn.Sequential(
            nn.Conv2d(effective_input_channel, expansion_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(expansion_channel),
            nn.ReLU6(inplace=True),
            nn.Conv2d(expansion_channel, expansion_channel, kernel_size=3, stride=self.stride, padding=1, groups=expansion_channel, bias=False),
            nn.BatchNorm2d(expansion_channel),
            nn.ReLU6(inplace=True),
            nn.Conv2d(expansion_channel, effective_output_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(effective_output_channel)
        )

    def forward(self, inputs):
        x = self.layers(inputs)
        if self.shortcut:
            x = x + inputs
        return x

class MobileNetV2(nn.Module):
    def __init__(self, output_size=10, alpha=1.0):
        super(MobileNetV2, self).__init__()
        self.conv0 = nn.Conv2d(3, int(32 * alpha), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(int(32 * alpha))
        self.bottlenecks = nn.Sequential(
            BaseBlock(32, 16, alpha, t=1), BaseBlock(16, 24, alpha), BaseBlock(24, 24, alpha),
            BaseBlock(24, 32, alpha), BaseBlock(32, 32, alpha), BaseBlock(32, 32, alpha),
            BaseBlock(32, 64, alpha, downsample=True), BaseBlock(64, 64, alpha), BaseBlock(64, 64, alpha),
            BaseBlock(64, 64, alpha), BaseBlock(64, 96, alpha), BaseBlock(96, 96, alpha),
            BaseBlock(96, 96, alpha), BaseBlock(96, 160, alpha, downsample=True), BaseBlock(160, 160, alpha),
            BaseBlock(160, 160, alpha), BaseBlock(160, 320, alpha)
        )
        last_conv_in = int(320 * alpha)
        last_conv_out = 1280
        self.conv1 = nn.Conv2d(last_conv_in, last_conv_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(last_conv_out)
        self.fc = nn.Linear(last_conv_out, output_size)

    def forward(self, inputs):
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace=True)
        x = self.bottlenecks(x)
        x = F.relu6(self.bn1(self.conv1(x)), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

def test(model, testloader, criterion, device):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return test_loss / len(testloader), 100 * correct / total


#pruning implementation
def prune_model_custom(model, pruning_ratio=0.5):
    print(f"==> Starting custom pruning with ratio: {pruning_ratio}")
    weights_to_prune = [module.weight.data for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear))]
    all_weights = torch.cat([w.view(-1) for w in weights_to_prune])
    threshold = torch.kthvalue(torch.abs(all_weights), int(len(all_weights) * pruning_ratio)).values.item()
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            mask = torch.abs(module.weight.data) > threshold
            module.weight.data.mul_(mask.float())
    return model


#Quantization

class QuantizationHook:
    def __init__(self, bits=8, calibration=False):
        self.bits, self.calibration, self.min_val, self.max_val = bits, calibration, None, None
    def __call__(self, module, module_in, module_out):
        if self.calibration:
            self.min_val = module_out.min() if self.min_val is None else min(self.min_val, module_out.min())
            self.max_val = module_out.max() if self.max_val is None else max(self.max_val, module_out.max())
            return
        q_min, q_max = 0, 2**self.bits - 1
        scale = (self.max_val - self.min_val) / (q_max - q_min) if self.max_val != self.min_val else 1e-8
        zero_point = q_min - self.min_val / scale
        quantized_out = torch.round(module_out / scale + zero_point).clamp(q_min, q_max)
        return (quantized_out - zero_point) * scale

def quantize_dequantize_weights(tensor, bits):
    q_max = 2**(bits - 1) - 1
    scale = tensor.abs().max() / q_max if q_max > 0 else 1.0
    return (tensor / scale).round().clamp(-q_max-1, q_max) * scale

def apply_quantization(model, wbits, abits, calibration_loader, device):
    q_model = copy.deepcopy(model)
    for module in q_model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.weight.data = quantize_dequantize_weights(module.weight.data, wbits)
    hooks = []
    for module in q_model.modules():
        if isinstance(module, nn.ReLU6):
            hook = QuantizationHook(bits=abits, calibration=True)
            hooks.append(hook)
            module.register_forward_hook(hook)
    print("  -> Calibrating activation ranges...")
    q_model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= 20: break
            q_model(images.to(device))
    for hook in hooks: hook.calibration = False
    print("  -> Calibration complete.")
    return q_model, hooks

def save_true_quantized_model(model, path, bits=8):
    if bits not in [8, 4]: raise ValueError("Only 8 and 4-bit saving supported.")
    quantized_state_dict = {}
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            q_max = 2**(bits - 1) - 1
            scale = param.abs().max() / q_max if q_max > 0 else 1.0
            quantized_tensor = (param.data / scale).round()
            if bits == 8:
                final_tensor = quantized_tensor.to(torch.int8)
            elif bits == 4:
                flattened = quantized_tensor.clamp(-8, 7).to(torch.int8).flatten()
                if flattened.numel() % 2 != 0: flattened = torch.cat([flattened, torch.tensor([0], dtype=torch.int8)])
                packed_tensor = torch.zeros(flattened.numel() // 2, dtype=torch.uint8)
                for i in range(packed_tensor.numel()):
                    packed_tensor[i] = ((flattened[2*i] & 0x0F) << 4) | (flattened[2*i + 1] & 0x0F)
                final_tensor = packed_tensor
            quantized_state_dict[name] = final_tensor
            quantized_state_dict[f'{name}_scale'] = scale
        else:
            quantized_state_dict[name] = param.data
    torch.save(quantized_state_dict, path)
    print(f"  -> Saved TRUE INT{bits} model to: {path}")



#Main function of the progran

if __name__ == "__main__":
    wandb.init()
    wbits, abits = wandb.config.wbits, wandb.config.abits
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define file paths (adjust if necessary)
    BASE_DIR = os.path.expanduser("")
    ORIGINAL_MODEL_PATH = os.path.join(BASE_DIR, 'weights_alpha_1.0_best.pkl')
    PRUNED_MODEL_PATH = os.path.join(BASE_DIR, 'mobilenet_pruned.pkl')
    DATA_PATH = os.path.join(BASE_DIR, 'data')
    
    # Data Loaders
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
    calibration_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    
    # Prepare Pruned Model (prune if it doesn't exist)
    if not os.path.exists(PRUNED_MODEL_PATH):
        print(f"Pruned model not found. Creating from: {ORIGINAL_MODEL_PATH}")
        original_model = MobileNetV2(output_size=10, alpha=1.0).to(device)
        original_model.load_state_dict(torch.load(ORIGINAL_MODEL_PATH, map_location=device))
        pruned_model = prune_model_custom(original_model, pruning_ratio=0.5)
        torch.save(pruned_model.state_dict(), PRUNED_MODEL_PATH)
    
    pruned_model = MobileNetV2(output_size=10, alpha=1.0).to(device)
    pruned_model.load_state_dict(torch.load(PRUNED_MODEL_PATH, map_location=device))
    
    # Apply Quantization for accuracy test
    quantized_model_sim, hooks = apply_quantization(pruned_model, wbits, abits, calibration_loader, device)
    
    # Test, Log, and Save
    criterion = nn.CrossEntropyLoss()
    _, acc_quantized = test(quantized_model_sim, testloader, criterion, device)
    wandb.log({"quantized_acc": acc_quantized})
    
    save_path = f'mobilenet_W{wbits}A{abits}.pkl'
    save_true_quantized_model(quantized_model_sim, save_path, bits=wbits)
    
    original_size = os.path.getsize(ORIGINAL_MODEL_PATH)
    final_size = os.path.getsize(save_path)
    compression_ratio = original_size / final_size
    
    wandb.log({
        "model_size_mb": final_size / 1e6,
        "compression_ratio": compression_ratio
    })

    artifact = wandb.Artifact('final-model', type='model', metadata=dict(wandb.config))
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)
    
    print(f"\n Run W{wbits}A{abits} Complete. Accuracy: {acc_quantized:.2f}%, Size: {final_size/1e6:.2f}MB, Ratio: {compression_ratio:.1f}x")
    wandb.finish()
