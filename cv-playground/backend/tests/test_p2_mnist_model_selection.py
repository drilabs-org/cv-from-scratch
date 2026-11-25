import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.registry import TaskExecutionError
from app.tasks import p2_mnist


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.cached_model = p2_mnist._MODEL
        self.cached_module = p2_mnist._MNIST
        self.cached_runtime_info = p2_mnist._MODEL_RUNTIME_INFO
        p2_mnist._MODEL = None
        p2_mnist._MNIST = None
        p2_mnist._MODEL_RUNTIME_INFO = None

    def tearDown(self):
        p2_mnist._MODEL = self.cached_model
        p2_mnist._MNIST = self.cached_module
        p2_mnist._MODEL_RUNTIME_INFO = self.cached_runtime_info

    def test_unset_directory_uses_bundled_model_without_fallback_warning(self):
        with patch.dict(os.environ, {}, clear=True):
            path, info = p2_mnist._resolve_model_weights()

        self.assertEqual(path, p2_mnist.BUNDLED_WEIGHTS)
        self.assertEqual(info["source"], "bundled")
        self.assertFalse(info["fallback"])

    def test_existing_custom_weights_take_precedence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = Path(temp_dir) / p2_mnist.MODEL_FILENAME
            custom_path.touch()
            with patch.dict(os.environ, {"CV_MODELS_DIR": temp_dir}, clear=True):
                path, info = p2_mnist._resolve_model_weights()

        self.assertEqual(path, custom_path)
        self.assertEqual(info["source"], "custom")
        self.assertFalse(info["fallback"])

    def test_missing_custom_weights_fall_back_to_bundled_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"CV_MODELS_DIR": temp_dir}, clear=True):
                path, info = p2_mnist._resolve_model_weights()

        self.assertEqual(path, p2_mnist.BUNDLED_WEIGHTS)
        self.assertEqual(info["source"], "bundled")
        self.assertTrue(info["fallback"])
        self.assertIn(p2_mnist.MODEL_FILENAME, info["message"])

    def test_invalid_custom_weights_raise_a_user_facing_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / p2_mnist.MODEL_FILENAME).touch()
            with (
                patch.dict(os.environ, {"CV_MODELS_DIR": temp_dir}, clear=True),
                patch(
                    "app.cvlib.mnist.load_model",
                    side_effect=RuntimeError("key mismatch"),
                ),
            ):
                with self.assertRaisesRegex(TaskExecutionError, "DigitCNN state_dict"):
                    p2_mnist._get_model()

        self.assertIsNone(p2_mnist._MODEL)


if __name__ == "__main__":
    unittest.main()
