"""Tests for the training model builder."""
from src.train import build_model


def test_build_model_shapes():
    """Model has correct input and output shapes."""
    model = build_model(4)
    assert model.input_shape == (None, 4)
    assert model.output_shape == (None, 1)


def test_model_compiled():
    """Model is compiled with an optimizer."""
    model = build_model(4)
    assert model.optimizer is not None
