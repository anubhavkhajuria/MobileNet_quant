import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# --- 1. IMPORT FROM YOUR SCRIPT ---
# We are importing the necessary components directly from your gem.py file.
try:
    from train import MobileNetV2, test
except ImportError:
    print("Error: Make sure 'train.py' is in the same directory as 'prune.py'")
    exit()

# --- 2. CUSTOM PRUNING IMPLEMENTATION ---

def calculate_sparsity(model):
    """
    Calculates the percentage of weights that are zero in a model.
    """
    total_weights = 0
    zero_weights = 0
    for param in model.parameters():
        if param.requires_grad:
            total_weights += param.nelement()
            zero_weights += torch.sum(param == 0).item()
    return 100 * zero_weights / total_weights


def prune_model_custom(model, pruning_ratio=0.5):
    """
    Performs magnitude-based unstructured pruning on a model from scratch.

    Args:
        model (nn.Module): The model to be pruned.
        pruning_ratio (float): The percentage of weights to prune (e.g., 0.5 for 50%).
    """
    print(f"==> Starting custom pruning with ratio: {pruning_ratio}")

    # 1. Gather all weights into a single list
    weights_to_prune = []
    for module in model.modules():
        # We only prune Conv2d and Linear layers
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weights_to_prune.append(module.weight.data.view(-1))

    # 2. Concatenate all weights and calculate the global threshold
    all_weights = torch.cat(weights_to_prune)
    threshold = torch.kthvalue(
        torch.abs(all_weights),
        int(len(all_weights) * pruning_ratio)
    ).values.item()

    print(f"Calculated global pruning threshold: {threshold:.4f}")

    # 3. Create and apply masks
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            
            # Create a binary mask (1 for weights to keep, 0 for weights to prune)
            mask = torch.abs(weight) > threshold
            
            # Apply the mask to the weights
            weight.mul_(mask.float())
            
            # This is the crucial step to make pruning permanent during training.
            # We register a "hook" that ensures the gradients for pruned weights are always zero.
            if hasattr(module.weight, 'hook'):
                module.weight.hook.remove() # Remove old hook if it exists

            module.weight.register_hook(lambda grad: grad * mask.float())

    print("==> Pruning complete.")
    return model


#THis is our main function which is going to call other and use their utilities.


if __name__ == "__main__":
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Path to your saved model (adjust if your alpha was different)
    MODEL_PATH = 'weights_alpha_1.0_best.pkl'
    PRUNING_PERCENTAGE = 0.5 # Prune 50% of the weights

    # --- Load Data ---
    print("==> Preparing CIFAR-10 test data...")
    cifar10_mean, cifar10_std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # --- Load Model ---
    print(f"==> Loading pre-trained model from: {MODEL_PATH}")
    model = MobileNetV2(output_size=10, alpha=1.0).to(device)
    try:
        # Handling DataParallel wrapper if the model was saved with it
        state_dict = torch.load(MODEL_PATH, map_location=device)
        if 'module' in list(state_dict.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please train the model first.")
        exit()

    criterion = nn.CrossEntropyLoss()

    # --- Evaluate Before Pruning ---
    sparsity_before = calculate_sparsity(model)
    print(f"\nSparsity before pruning: {sparsity_before:.2f}%")
    print("Testing original model accuracy...")
    _, acc_before = test(model, testloader, criterion, device)
    print(f"  -> Accuracy before pruning: {acc_before:.2f}%")

    # --- Prune the Model ---
    model = prune_model_custom(model, pruning_ratio=PRUNING_PERCENTAGE)

    # --- Evaluate After Pruning ---
    sparsity_after = calculate_sparsity(model)
    print(f"\nSparsity after pruning: {sparsity_after:.2f}%")
    print("Testing pruned model accuracy...")
    _, acc_after = test(model, testloader, criterion, device)
    print(f"  -> Accuracy after pruning : {acc_after:.2f}%")
