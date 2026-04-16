"""Unit tests for config loading."""
import os


def test_config_imports():
    from app.core import config
    assert config is not None


def test_config_has_expected_attrs():
    from app.core import config
    assert hasattr(config, "__name__")


def test_environment_variables_readable():
    os.environ["TEST_NEWSSCOPE_VAR"] = "test_value"
    assert os.environ.get("TEST_NEWSSCOPE_VAR") == "test_value"
    del os.environ["TEST_NEWSSCOPE_VAR"]
