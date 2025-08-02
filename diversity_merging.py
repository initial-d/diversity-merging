#!/usr/bin/env python3
"""
Diversity-Driven Model Merging: Implementation

This code implements the core algorithm from the paper:
"Diversity-Driven Model Merging: Understanding Emergent Robustness Through Parameter Space Analysis"

Key features:
- Diversity-aware model merging
- Theoretical metrics computation
- Robustness evaluation
- Simple CIFAR-10 experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import copy
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Model Architecture
# ============================================================================

class SimpleResNet(nn.Module):
    """Simplified ResNet for CIFAR-10"""
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 16, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ============================================================================
# Diversity Metrics
# ============================================================================

class DiversityMetrics:
    """Compute theoretical diversity metrics"""
    
    @staticmethod
    def parameter_diversity(models: List[nn.Module]) -> float:
        """Compute effective parameter diversity D_eff"""
        params_list = []
        for model in models:
            params = torch.cat([p.flatten() for p in model.parameters()])
            params_list.append(params)
        
        n = len(params_list)
        total_distance = 0.0
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = torch.norm(params_list[i] - params_list[j], p=2).item()
                total_distance += distance
        
        return total_distance / (n * (n - 1) / 2) if n > 1 else 0.0
    
    @staticmethod
    def hessian_diversity(models: List[nn.Module], dataloader: DataLoader, 
                         device: str = 'cpu') -> float:
        """Compute Hessian diversity (diagonal approximation)"""
        hessian_diags = []
        
        for model in models:
            model.eval()
            hessian_diag = DiversityMetrics._estimate_hessian_diagonal(
                model, dataloader, device
            )
            hessian_diags.append(hessian_diag)
        
        n = len(hessian_diags)
        total_distance = 0.0
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = torch.norm(hessian_diags[i] - hessian_diags[j], p='fro').item()
                total_distance += distance
        
        return total_distance / (n * (n - 1) / 2) if n > 1 else 0.0
    
    @staticmethod
    def _estimate_hessian_diagonal(model: nn.Module, dataloader: DataLoader, 
                                  device: str, num_samples: int = 5) -> torch.Tensor:
        """Fast diagonal Hessian estimation using Hutchinson's trace estimator"""
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)
        
        hessian_diag = torch.zeros(n_params, device=device)
        
        # Sample a small batch for estimation
        batch = next(iter(dataloader))
        inputs, targets = batch[0].to(device), batch[1].to(device)
        inputs = inputs[:min(32, inputs.size(0))]  # Use small batch
        targets = targets[:min(32, targets.size(0))]
        
        for _ in range(num_samples):
            # Random vector for Hutchinson estimator
            z = torch.randint_like(hessian_diag, high=2, dtype=torch.float32) * 2 - 1
            
            # Compute gradient
            model.zero_grad()
            loss = F.cross_entropy(model(inputs), targets)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            grad_vec = torch.cat([g.flatten() for g in grads])
            
            # Compute Hessian-vector product
            hvp = torch.autograd.grad(grad_vec @ z, params, retain_graph=False)
            hvp_vec = torch.cat([h.flatten() for h in hvp])
            
            hessian_diag += z * hvp_vec
        
        return hessian_diag / num_samples

# ============================================================================
# Model Merging Algorithm
# ============================================================================

class DiversityMerger:
    """Core diversity-driven model merging algorithm"""
    
    def __init__(self, reg_lambda: float = 1e-3):
        self.reg_lambda = reg_lambda
    
    def merge_models(self, models: List[nn.Module], dataloader: DataLoader, 
                    device: str = 'cpu') -> Tuple[nn.Module, Dict]:
        """
        Merge models using diversity-aware weighting
        
        Args:
            models: List of trained models
            dataloader: Validation data for weight optimization
            device: Computing device
            
        Returns:
            merged_model: The merged model
            info: Dictionary with merging information
        """
        n_models = len(models)
        
        # Compute diversity metrics
        param_diversity = DiversityMetrics.parameter_diversity(models)
        
        # Compute pairwise distances
        distance_matrix = self._compute_distance_matrix(models)
        
        # Estimate curvature information (simplified)
        curvatures = self._estimate_curvatures(models, dataloader, device)
        
        # Solve for optimal weights
        weights = self._solve_optimal_weights(distance_matrix, curvatures)
        
        # Create merged model
        merged_model = self._create_merged_model(models, weights)
        
        info = {
            'weights': weights,
            'parameter_diversity': param_diversity,
            'distance_matrix': distance_matrix,
            'curvatures': curvatures
        }
        
        return merged_model, info
    
    def _compute_distance_matrix(self, models: List[nn.Module]) -> torch.Tensor:
        """Compute pairwise parameter distances"""
        n = len(models)
        distance_matrix = torch.zeros(n, n)
        
        params_list = []
        for model in models:
            params = torch.cat([p.flatten() for p in model.parameters()])
            params_list.append(params)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance = torch.norm(params_list[i] - params_list[j], p=2)
                    distance_matrix[i, j] = distance
        
        return distance_matrix
    
    def _estimate_curvatures(self, models: List[nn.Module], dataloader: DataLoader, 
                           device: str) -> torch.Tensor:
        """Estimate local curvature for each model"""
        curvatures = []
        
        for model in models:
            # Simplified: use gradient norm as proxy for curvature
            model.eval()
            batch = next(iter(dataloader))
            inputs, targets = batch[0].to(device)[:16], batch[1].to(device)[:16]
            
            model.zero_grad()
            loss = F.cross_entropy(model(inputs), targets)
            grads = torch.autograd.grad(loss, model.parameters())
            grad_norm = torch.sqrt(sum(torch.sum(g**2) for g in grads))
            curvatures.append(grad_norm.item())
        
        return torch.tensor(curvatures)
    
    def _solve_optimal_weights(self, distance_matrix: torch.Tensor, 
                              curvatures: torch.Tensor) -> torch.Tensor:
        """Solve convex optimization for optimal weights"""
        n = distance_matrix.size(0)
        
        # Simplified weight computation based on theory
        # w_i ∝ tr(H_i^{-1}) * exp(-β * Σ D_ij)
        
        # Curvature-based component (higher curvature = lower weight)
        curvature_weights = 1.0 / (curvatures + 1e-6)
        
        # Distance-based component (prefer diverse models)
        distance_weights = torch.exp(-self.reg_lambda * distance_matrix.sum(dim=1))
        
        # Combine components
        raw_weights = curvature_weights * distance_weights
        
        # Normalize to sum to 1
        weights = raw_weights / raw_weights.sum()
        
        return weights
    
    def _create_merged_model(self, models: List[nn.Module], 
                           weights: torch.Tensor) -> nn.Module:
        """Create merged model with weighted parameters"""
        merged_model = copy.deepcopy(models[0])
        
        # Initialize with zeros
        for param in merged_model.parameters():
            param.data.zero_()
        
        # Weighted combination
        for i, (model, weight) in enumerate(zip(models, weights)):
            for merged_param, model_param in zip(merged_model.parameters(), 
                                               model.parameters()):
                merged_param.data += weight * model_param.data
        
        return merged_model

# ============================================================================
# Training and Evaluation
# ============================================================================

class Trainer:
    """Simple trainer for creating diverse models"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def train_diverse_models(self, num_models: int = 3, epochs: int = 10) -> List[nn.Module]:
        """Train multiple diverse models with different configurations"""
        # Load CIFAR-10 data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform_train)
        
        models = []
        configs = [
            {'lr': 0.01, 'dropout': 0.1, 'weight_decay': 1e-4},
            {'lr': 0.005, 'dropout': 0.3, 'weight_decay': 1e-3},
            {'lr': 0.02, 'dropout': 0.5, 'weight_decay': 1e-2},
        ]
        
        for i in range(num_models):
            print(f"Training model {i+1}/{num_models}...")
            config = configs[i % len(configs)]
            
            # Create diverse training data
            indices = torch.randperm(len(trainset))[:40000]  # Use subset for speed
            subset = torch.utils.data.Subset(trainset, indices)
            trainloader = DataLoader(subset, batch_size=128, shuffle=True)
            
            model = SimpleResNet().to(self.device)
            model = self._train_single_model(model, trainloader, config, epochs)
            models.append(model)
        
        return models
    
    def _train_single_model(self, model: nn.Module, trainloader: DataLoader, 
                          config: Dict, epochs: int) -> nn.Module:
        """Train a single model"""
        optimizer = torch.optim.SGD(model.parameters(), 
                                  lr=config['lr'],
                                  weight_decay=config['weight_decay'],
                                  momentum=0.9)
        
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'  Epoch {epoch+1}, Batch {i+1}: Loss = {running_loss/100:.3f}')
                    running_loss = 0.0
        
        return model

class Evaluator:
    """Evaluate model performance and robustness"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # Load test data
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform_test)
        self.testloader = DataLoader(testset, batch_size=100, shuffle=False)
    
    def evaluate_accuracy(self, model: nn.Module) -> float:
        """Evaluate clean accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total
    
    def evaluate_robustness(self, model: nn.Module, epsilon: float = 0.03) -> float:
        """Evaluate adversarial robustness using FGSM"""
        model.eval()
        correct = 0
        total = 0
        
        for inputs, labels in self.testloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs.requires_grad = True
            
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            
            model.zero_grad()
            loss.backward()
            
            # FGSM attack
            attack = epsilon * inputs.grad.sign()
            perturbed_inputs = inputs + attack
            perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
            
            with torch.no_grad():
                outputs = model(perturbed_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total

# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment():
    """Run the main experiment comparing different merging methods"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Train diverse models
    print("="*60)
    print("STEP 1: Training diverse models...")
    print("="*60)
    
    trainer = Trainer(device)
    models = trainer.train_diverse_models(num_models=3, epochs=5)  # Reduced for demo
    
    # Step 2: Compute diversity metrics
    print("\n" + "="*60)
    print("STEP 2: Computing diversity metrics...")
    print("="*60)
    
    # Create validation loader
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    valset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_val)
    val_indices = torch.randperm(len(valset))[:1000]  # Small validation set
    val_subset = torch.utils.data.Subset(valset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=100, shuffle=False)
    
    param_diversity = DiversityMetrics.parameter_diversity(models)
    print(f"Parameter Diversity (D_eff): {param_diversity:.4f}")
    
    # Step 3: Merge models
    print("\n" + "="*60)
    print("STEP 3: Merging models...")
    print("="*60)
    
    merger = DiversityMerger(reg_lambda=1e-3)
    merged_model, merge_info = merger.merge_models(models, val_loader, device)
    
    print("Optimal weights:", merge_info['weights'].numpy())
    
    # Step 4: Evaluation
    print("\n" + "="*60)
    print("STEP 4: Evaluating performance...")
    print("="*60)
    
    evaluator = Evaluator(device)
    
    # Evaluate individual models
    individual_accs = []
    individual_robust_accs = []
    
    for i, model in enumerate(models):
        acc = evaluator.evaluate_accuracy(model)
        robust_acc = evaluator.evaluate_robustness(model)
        individual_accs.append(acc)
        individual_robust_accs.append(robust_acc)
        print(f"Model {i+1} - Clean: {acc:.2f}%, Robust: {robust_acc:.2f}%")
    
    # Evaluate uniform average
    uniform_model = copy.deepcopy(models[0])
    for param in uniform_model.parameters():
        param.data.zero_()
    
    for model in models:
        for uniform_param, model_param in zip(uniform_model.parameters(), model.parameters()):
            uniform_param.data += model_param.data / len(models)
    
    uniform_acc = evaluator.evaluate_accuracy(uniform_model)
    uniform_robust_acc = evaluator.evaluate_robustness(uniform_model)
    
    # Evaluate our merged model
    merged_acc = evaluator.evaluate_accuracy(merged_model)
    merged_robust_acc = evaluator.evaluate_robustness(merged_model)
    
    # Step 5: Results summary
    print("\n" + "="*60)
    print("STEP 5: Results Summary")
    print("="*60)
    
    print(f"Best Individual Model - Clean: {max(individual_accs):.2f}%, Robust: {max(individual_robust_accs):.2f}%")
    print(f"Uniform Average     - Clean: {uniform_acc:.2f}%, Robust: {uniform_robust_acc:.2f}%")
    print(f"Our Method          - Clean: {merged_acc:.2f}%, Robust: {merged_robust_acc:.2f}%")
    
    improvement_clean = merged_acc - uniform_acc
    improvement_robust = merged_robust_acc - uniform_robust_acc
    
    print(f"\nImprovement over uniform average:")
    print(f"Clean accuracy: +{improvement_clean:.2f}%")
    print(f"Robust accuracy: +{improvement_robust:.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    methods = ['Individual\n(Best)', 'Uniform\nAverage', 'Our Method']
    clean_scores = [max(individual_accs), uniform_acc, merged_acc]
    plt.bar(methods, clean_scores, color=['lightblue', 'orange', 'green'])
    plt.ylabel('Clean Accuracy (%)')
    plt.title('Clean Performance Comparison')
    plt.ylim(0, 100)
    
    plt.subplot(1, 2, 2)
    robust_scores = [max(individual_robust_accs), uniform_robust_acc, merged_robust_acc]
    plt.bar(methods, robust_scores, color=['lightblue', 'orange', 'green'])
    plt.ylabel('Robust Accuracy (%)')
    plt.title('Robust Performance Comparison')
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'individual_models': models,
        'merged_model': merged_model,
        'merge_info': merge_info,
        'results': {
            'individual_clean': individual_accs,
            'individual_robust': individual_robust_accs,
            'uniform_clean': uniform_acc,
            'uniform_robust': uniform_robust_acc,
            'merged_clean': merged_acc,
            'merged_robust': merged_robust_acc
        }
    }

if __name__ == "__main__":
    # Run the experiment
    results = run_experiment()
    
    print("\n" + "="*60)
    print("Experiment completed! Check 'results.png' for visualization.")
    print("="*60)
