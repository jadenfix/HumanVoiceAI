"""
Model interpretability using SHAP (SHapley Additive exPlanations).
"""

import numpy as np
import torch
import shap
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib

# Use non-interactive backend for matplotlib
matplotlib.use("Agg")


class SERExplainer:
    """SHAP explainer for Speech Emotion Recognition models."""

    def __init__(
        self,
        model: torch.nn.Module,
        background_data: np.ndarray,
        class_names: List[str],
        device: str = "cpu",
    ):
        """Initialize the SHAP explainer.

        Args:
            model: Trained SER model
            background_data: Background data for SHAP (n_samples, n_features)
            class_names: List of emotion class names
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model = model
        self.class_names = class_names
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Convert background data to tensor
        background_tensor = torch.FloatTensor(background_data).to(self.device)

        # Define a wrapper function for the model
        def model_wrapper(x):
            x_tensor = torch.FloatTensor(x).to(self.device)
            with torch.no_grad():
                outputs = self.model(x_tensor)
            return outputs.cpu().numpy()

        # Initialize SHAP explainer
        self.explainer = shap.DeepExplainer(
            model=model_wrapper,
            data=background_tensor[:100],  # Use first 100 samples as background
        )

    def explain(
        self,
        input_data: np.ndarray,
        class_idx: Optional[int] = None,
        nsamples: int = 100,
    ) -> Dict[str, Any]:
        """Generate SHAP explanations for the input data.

        Args:
            input_data: Input data (n_samples, n_features)
            class_idx: Class index to explain (None for predicted class)
            nsamples: Number of samples to use for approximation

        Returns:
            Dictionary containing SHAP values and visualization
        """
        # Convert input to tensor if needed
        if not isinstance(input_data, torch.Tensor):
            input_tensor = torch.FloatTensor(input_data)
        else:
            input_tensor = input_data

        # Get model predictions
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        # If no class specified, use predicted class
        if class_idx is None:
            class_idx = pred_class

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(
            input_data, check_additivity=False, nsamples=nsamples
        )

        # Get SHAP values for the target class
        class_shap_values = shap_values[class_idx]

        # Create visualization
        plt.figure(figsize=(10, 4))
        shap.summary_plot(
            class_shap_values,
            input_data,
            feature_names=[f"MFCC_{i}" for i in range(input_data.shape[1])],
            class_names=self.class_names,
            class_inds=class_idx,
            show=False,
        )
        plt.tight_layout()

        return {
            "shap_values": class_shap_values,
            "predicted_class": pred_class,
            "predicted_prob": probs[0, pred_class].item(),
            "target_class": class_idx,
            "target_prob": probs[0, class_idx].item(),
            "class_names": self.class_names,
            "figure": plt.gcf(),
        }

    def plot_heatmap(
        self,
        shap_values: np.ndarray,
        feature_names: Optional[List[str]] = None,
        title: str = "SHAP Values Heatmap",
    ) -> plt.Figure:
        """Plot a heatmap of SHAP values.

        Args:
            shap_values: SHAP values (n_samples, n_features)
            feature_names: List of feature names
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if feature_names is None:
            feature_names = [f"MFCC_{i}" for i in range(shap_values.shape[1])]

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(shap_values, aspect="auto", cmap="viridis")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("SHAP value", rotation=-90, va="bottom")

        # Set ticks
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("Time steps")
        ax.set_title(title)

        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add grid
        ax.set_xticks(np.arange(-0.5, shap_values.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(feature_names), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        plt.tight_layout()
        return fig

    def plot_summary(
        self,
        shap_values: np.ndarray,
        feature_names: Optional[List[str]] = None,
        title: str = "SHAP Summary Plot",
    ) -> plt.Figure:
        """Create a summary plot of SHAP values.

        Args:
            shap_values: SHAP values (n_samples, n_features)
            feature_names: List of feature names
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if feature_names is None:
            feature_names = [f"MFCC_{i}" for i in range(shap_values.shape[1])]

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, feature_names=feature_names, show=False, plot_type="bar"
        )
        plt.title(title)
        plt.tight_layout()
        return fig


def test_explainer():
    """Test function for the SHAP explainer."""
    print("Testing SHAP explainer...")

    # Create a dummy model for testing
    class DummyModel(torch.nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.fc = torch.nn.Linear(input_dim, num_classes)

        def forward(self, x):
            return self.fc(x)

    # Test parameters
    input_dim = 20
    num_classes = 5
    num_samples = 100

    # Create dummy data
    background_data = np.random.randn(50, input_dim)
    test_data = np.random.randn(10, input_dim)

    # Initialize model and explainer
    model = DummyModel(input_dim, num_classes)
    class_names = [f"Class_{i}" for i in range(num_classes)]
    explainer = SERExplainer(model, background_data, class_names)

    # Generate explanations
    explanation = explainer.explain(test_data[0:1])

    # Plot results
    plt.figure(figsize=(10, 4))
    shap.summary_plot(
        explanation["shap_values"],
        test_data[0:1],
        feature_names=[f"Feature_{i}" for i in range(input_dim)],
        show=False,
    )
    plt.title("SHAP Values")
    plt.tight_layout()

    print("Test completed. Close the plot to continue.")
    plt.show()


if __name__ == "__main__":
    test_explainer()
