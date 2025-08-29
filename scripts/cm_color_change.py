import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# put your own confusion matrix values
confusion_data = np.array([
    [302, 13, 0, 1],  # Cavities (actual)
    [11, 401, 3, 0],  # Damage (actual)
    [0, 0, 61, 0],  # Infection (actual)
    [1, 0, 0, 18]  # Wisdom (actual)
])

# Define class labels
class_labels = ['Cavities', 'Damage', 'Infection', 'Wisdom']

# Create the confusion matrix plot
plt.figure(figsize=(10, 8))

# Use blue-white color scheme
# 'Blues' colormap goes from white to dark blue
sns.heatmap(confusion_data,
            annot=True,  # Show numbers
            fmt='d',  # Integer format
            cmap='Blues',  # Blue-white theme
            xticklabels=class_labels,  # X-axis labels
            yticklabels=class_labels,  # Y-axis labels
            cbar_kws={'label': 'Count'},  # Color bar label
            annot_kws={'size': 16,  # Make numbers larger
                       'weight': 'bold',  # Make numbers bold
                       'color': 'black'},  # Text color
            linewidths=0.5,  # Add grid lines
            linecolor='gray')  # Grid line color

# Customize the plot
plt.title('Confusion Matrix (Test)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Predicted', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=14, fontweight='bold')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Optional: Save the figure
plt.savefig('confusion_matrix_dental.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()


# Optional: Calculate and print accuracy metrics
def calculate_metrics(cm):
    """Calculate precision, recall, and F1-score for each class"""
    n_classes = cm.shape[0]

    print("\nPerformance Metrics:")
    print("-" * 50)

    for i, class_name in enumerate(class_labels):
        # True Positives, False Positives, False Negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{class_name:10} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    # Overall accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"\nOverall Accuracy: {accuracy:.3f}")


calculate_metrics(confusion_data)