import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dense import DenseNetwork, Pooling
from src.training.modules import PipelineModule


class TestPooling:
    """Tests for Pooling layer."""

    def test_mean_pooling(self):
        """Test mean pooling."""
        pool = Pooling(pool_type='mean', dim=1)
        x = torch.ones(2, 10, 5)
        result = pool(x)
        assert result.shape == (2, 5)
        assert torch.allclose(result, torch.ones(2, 5))

    def test_max_pooling(self):
        """Test max pooling."""
        pool = Pooling(pool_type='max', dim=1)
        x = torch.arange(10).unsqueeze(0).unsqueeze(-1).float()  # (1, 10, 1)
        result = pool(x)
        assert result.shape == (1, 1)
        assert result.item() == 9.0

    def test_sum_pooling(self):
        """Test sum pooling."""
        pool = Pooling(pool_type='sum', dim=1)
        x = torch.ones(2, 10, 5)
        result = pool(x)
        assert result.shape == (2, 5)
        assert torch.allclose(result, torch.full((2, 5), 10.0))

    def test_invalid_pool_type(self):
        """Test that invalid pool type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid pool type"):
            Pooling(pool_type='invalid', dim=1)

    def test_with_nonlinearity(self):
        """Test pooling with nonlinearity."""
        pool = Pooling(pool_type='mean', dim=1, nonlinearity='relu')
        x = torch.full((2, 10, 5), -1.0)
        result = pool(x)
        assert torch.allclose(result, torch.zeros(2, 5))  # ReLU zeros negative


class TestDenseNetwork:
    """Tests for DenseNetwork."""

    def test_forward_no_hidden(self):
        """Test forward pass with no hidden layers."""
        net = DenseNetwork(in_size=10, out_size=2)
        x = torch.randn(5, 10)
        result = net(x)
        assert result.shape == (5, 2)

    def test_forward_with_hidden(self):
        """Test forward pass with hidden layers."""
        net = DenseNetwork(in_size=10, out_size=2, hidden_sizes=[32, 16])
        x = torch.randn(5, 10)
        result = net(x)
        assert result.shape == (5, 2)

    def test_forward_with_pooling(self):
        """Test forward pass with pooling."""
        net = DenseNetwork(
            in_size=10,
            out_size=1,
            pooling={'type': 'mean', 'dim': 1}
        )
        x = torch.randn(5, 20, 10)  # (batch, seq, features)
        result = net(x)
        # After forward: (5, 20, 1) -> squeeze -> (5, 20) -> pool dim 1 -> (5,)
        assert result.shape == (5,)

    def test_forward_with_dropout(self):
        """Test that dropout is applied during training."""
        net = DenseNetwork(in_size=10, out_size=2, hidden_sizes=[32], dropout=0.5)
        net.train()
        x = torch.randn(5, 10)

        # Multiple forward passes should give different results due to dropout
        results = [net(x) for _ in range(10)]
        all_same = all(torch.equal(results[0], r) for r in results[1:])
        assert not all_same, "Dropout should cause variation in outputs"

    def test_eval_mode_deterministic(self):
        """Test that eval mode gives deterministic results."""
        net = DenseNetwork(in_size=10, out_size=2, hidden_sizes=[32], dropout=0.5)
        net.eval()
        x = torch.randn(5, 10)

        result1 = net(x)
        result2 = net(x)
        assert torch.equal(result1, result2)

    def test_len(self):
        """Test __len__ returns number of layers."""
        net = DenseNetwork(in_size=10, out_size=2, hidden_sizes=[32, 16])
        assert len(net) == 3  # input->32, 32->16, 16->output

    def test_getitem(self):
        """Test __getitem__ returns correct layer."""
        net = DenseNetwork(in_size=10, out_size=2)
        layer = net[0]
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 10
        assert layer.out_features == 2


class TestPipelineModule:
    """Tests for PipelineModule."""

    def test_init_with_model(self):
        """Test initialization with a simple model."""
        model = nn.Linear(10, 2)
        module = PipelineModule(model=model)
        assert module.model is model

    def test_init_requires_loss_in_objectives(self):
        """Test that objectives must contain 'loss' key."""
        model = nn.Linear(10, 2)
        objectives = {'accuracy': nn.MSELoss()}  # No 'loss' key
        with pytest.raises(ValueError, match="loss"):
            PipelineModule(model=model, objectives=objectives)

    def test_init_with_valid_objectives(self):
        """Test initialization with valid objectives."""
        model = nn.Linear(10, 2)
        objectives = {
            'loss': nn.MSELoss(),
            'metric': nn.L1Loss(),
        }
        module = PipelineModule(model=model, objectives=objectives)
        assert 'loss' in module.objectives
        assert 'metric' in module.objectives

    def test_forward_delegates_to_model(self):
        """Test that forward() calls the underlying model."""
        model = nn.Linear(10, 2)
        module = PipelineModule(model=model)
        x = torch.randn(5, 10)

        result = module(x)
        expected = model(x)
        assert torch.equal(result, expected)

    def test_hooks_list_initialized(self):
        """Test that hooks list is initialized."""
        model = nn.Linear(10, 2)
        module = PipelineModule(model=model)
        assert hasattr(module, 'hooks')
