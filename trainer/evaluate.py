"""
Evaluation utilities for cat face recognition with Triplet Loss.
Includes metrics computation, visualization, and reporting.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, f1_score, precision_score, 
    recall_score, accuracy_score
)
from sklearn.manifold import TSNE
from scipy import stats

@torch.no_grad()
def generate_embeddings(model, dataset, device, max_samples=5000):
    """
    Generate embeddings for validation dataset.
    
    Args:
        model: Trained embedding network
        dataset: CatTripletDataset instance
        device: torch device (cuda/cpu)
        max_samples: Maximum number of triplets to process
        
    Returns:
        embeddings: numpy array of shape (N, embedding_dim)
        cat_ids: numpy array of cat IDs corresponding to embeddings
    """
    model.eval()
    
    embeddings = []
    cat_ids = []
    
    indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
    
    print(f"Generating embeddings for {len(indices)} samples...")
    
    for idx in tqdm(indices, desc="Generating embeddings"):
        anchor, positive, negative = dataset[idx]
        
        # Get embeddings for all three images
        anchor_emb = model(anchor.unsqueeze(0).to(device)).cpu().numpy()
        pos_emb = model(positive.unsqueeze(0).to(device)).cpu().numpy()
        neg_emb = model(negative.unsqueeze(0).to(device)).cpu().numpy()
        
        # Get corresponding cat IDs
        anchor_idx, pos_idx, neg_idx = dataset.triplets[idx]
        anchor_cat = dataset.labels[anchor_idx]
        pos_cat = dataset.labels[pos_idx]
        neg_cat = dataset.labels[neg_idx]
        
        embeddings.extend([anchor_emb[0], pos_emb[0], neg_emb[0]])
        cat_ids.extend([anchor_cat, pos_cat, neg_cat])
    
    embeddings = np.array(embeddings)
    cat_ids = np.array(cat_ids)
    
    print(f"✓ Generated embeddings shape: {embeddings.shape}")
    print(f"✓ Unique cats: {len(np.unique(cat_ids))}")
    
    return embeddings, cat_ids

@torch.no_grad()
def compute_distances(model, dataset, device, num_samples=2000):
    """
    Compute pairwise distances for verification task.
    
    Args:
        model: Trained embedding network
        dataset: CatTripletDataset instance
        device: torch device
        num_samples: Number of triplets to sample
        
    Returns:
        distances: Array of distances
        labels: Array of labels (1=same cat, 0=different cat)
    """
    model.eval()
    
    distances = []
    labels = []
    
    print(f"Computing distances for {num_samples} samples...")
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in tqdm(indices, desc="Computing distances"):
        anchor, positive, negative = dataset[idx]
        
        # Move to device
        anchor = anchor.unsqueeze(0).to(device)
        positive = positive.unsqueeze(0).to(device)
        negative = negative.unsqueeze(0).to(device)
        
        # Get embeddings
        anchor_emb = model(anchor)
        pos_emb = model(positive)
        neg_emb = model(negative)
        
        # Compute Euclidean distances
        pos_dist = torch.norm(anchor_emb - pos_emb, p=2, dim=1).item()
        neg_dist = torch.norm(anchor_emb - neg_emb, p=2, dim=1).item()
        
        distances.extend([pos_dist, neg_dist])
        labels.extend([1, 0])  # 1=same cat, 0=different cat
    
    distances = np.array(distances)
    labels = np.array(labels)
    
    print(f"✓ Positive pairs: {np.sum(labels == 1)}")
    print(f"✓ Negative pairs: {np.sum(labels == 0)}")
    
    return distances, labels

def compute_metrics(distances, true_labels):
    """
    Compute verification metrics with optimal threshold.
    
    Args:
        distances: Array of pairwise distances
        true_labels: Ground truth labels (1=same, 0=different)
        
    Returns:
        dict with all metrics and optimal threshold
    """
    # ROC Curve (negative distances because lower = more similar)
    fpr, tpr, thresholds = roc_curve(true_labels, -distances)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold using Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = -thresholds[optimal_idx]
    
    # Predictions using optimal threshold
    predictions = (distances < optimal_threshold).astype(int)
    
    # Compute classification metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)
    
    # Distance statistics
    pos_distances = distances[true_labels == 1]
    neg_distances = distances[true_labels == 0]
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'pos_dist_mean': pos_distances.mean(),
        'pos_dist_std': pos_distances.std(),
        'neg_dist_mean': neg_distances.mean(),
        'neg_dist_std': neg_distances.std(),
        'distance_separation': neg_distances.mean() - pos_distances.mean()
    }
    
    return metrics

def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=['Different Cat', 'Same Cat'],
                yticklabels=['Different Cat', 'Same Cat'])
    plt.title('Confusion Matrix - Cat Face Verification', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved as '{save_path}'")

def plot_roc_curve(fpr, tpr, roc_auc, optimal_threshold, optimal_idx, save_path='roc_curve.png'):
    """Plot and save ROC curve."""
    plt.figure(figsize=(10, 8))
    
    plt.plot(fpr, tpr, color='darkorange', lw=3, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.50)')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=150, 
                label=f'Optimal Threshold = {optimal_threshold:.3f}', 
                zorder=5, edgecolor='black', linewidth=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curve - Cat Face Verification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curve saved as '{save_path}'")


def plot_distance_distribution(distances, true_labels, optimal_threshold, save_path='distance_distribution.png'):
    """Plot histogram and KDE of distance distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    pos_distances = distances[true_labels == 1]
    neg_distances = distances[true_labels == 0]
    
    # Histogram
    axes[0].hist(pos_distances, bins=50, alpha=0.7, label='Same Cat (Positive)', 
                 color='green', edgecolor='black', linewidth=1.2)
    axes[0].hist(neg_distances, bins=50, alpha=0.7, label='Different Cat (Negative)', 
                 color='red', edgecolor='black', linewidth=1.2)
    axes[0].axvline(optimal_threshold, color='blue', linestyle='--', linewidth=2.5, 
                    label=f'Threshold = {optimal_threshold:.3f}')
    axes[0].set_xlabel('Euclidean Distance', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distance Distribution - Histogram', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # KDE Plot
    pos_kde = stats.gaussian_kde(pos_distances)
    neg_kde = stats.gaussian_kde(neg_distances)
    
    x_range = np.linspace(0, max(distances), 300)
    axes[1].fill_between(x_range, pos_kde(x_range), alpha=0.5, label='Same Cat', color='green')
    axes[1].fill_between(x_range, neg_kde(x_range), alpha=0.5, label='Different Cat', color='red')
    axes[1].axvline(optimal_threshold, color='blue', linestyle='--', linewidth=2.5, 
                    label=f'Threshold = {optimal_threshold:.3f}')
    axes[1].set_xlabel('Euclidean Distance', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Distance Distribution - Density Plot', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Distance distribution saved as '{save_path}'")


def plot_tsne(embeddings, cat_ids, n_top_cats=20, save_prefix='tsne_embeddings'):
    """
    Generate and plot t-SNE visualization.
    
    Args:
        embeddings: Array of embeddings
        cat_ids: Array of cat IDs
        n_top_cats: Number of top cats to highlight
        save_prefix: Prefix for saved files
    """
    # Sample if too many
    max_samples = 1000
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        cat_ids = cat_ids[indices]
    
    print(f"Running t-SNE on {len(embeddings)} samples...")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_results = tsne.fit_transform(embeddings)
    
    print("✓ t-SNE completed!")
    
    # Plot 1: Top N cats with different colors
    cat_counts = Counter(cat_ids)
    top_cats = [cat for cat, count in cat_counts.most_common(n_top_cats)]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for cat_id in top_cats:
        mask = cat_ids == cat_id
        ax.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                   alpha=0.7, s=40, label=f'Cat {cat_id}', 
                   edgecolor='black', linewidth=0.5)
    
    # Other cats in gray
    other_mask = ~np.isin(cat_ids, top_cats)
    ax.scatter(tsne_results[other_mask, 0], tsne_results[other_mask, 1], 
               alpha=0.3, s=15, color='gray', label='Other cats')
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=13)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=13)
    ax.set_title(f't-SNE Visualization (Top {n_top_cats} Cats)', 
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_top{n_top_cats}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ t-SNE plot (top {n_top_cats}) saved")
    
    # Plot 2: All cats with color gradient
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                         c=cat_ids, cmap='Spectral', alpha=0.6, s=25, 
                         edgecolor='black', linewidth=0.3)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=13)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=13)
    ax.set_title('t-SNE Visualization - All Cats', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, label='Cat ID')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ t-SNE plot (all cats) saved")

def evaluate_model(model, dataset, device, config=None, num_distance_samples=2000, 
                   num_embedding_samples=3000):
    """
    Run comprehensive evaluation on trained model.
    
    Args:
        model: Trained embedding network
        dataset: CatTripletDataset for validation
        device: torch device
        config: Training configuration dict (optional)
        num_distance_samples: Samples for distance computation
        num_embedding_samples: Samples for embedding generation
        
    Returns:
        metrics: Dictionary with all computed metrics
    """
    print("\n" + "="*70)
    print("STARTING COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # 1. Generate embeddings
    print("\n[1/5] Generating embeddings...")
    embeddings, cat_ids = generate_embeddings(model, dataset, device, num_embedding_samples)
    
    # 2. Compute distances
    print("\n[2/5] Computing distances...")
    distances, labels = compute_distances(model, dataset, device, num_distance_samples)
    
    # 3. Compute metrics
    print("\n[3/5] Computing metrics...")
    metrics = compute_metrics(distances, labels)
    
    # Print metrics
    print(f"\n{'='*60}")
    print("VERIFICATION METRICS")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    print(f"{'='*60}")
    
    # 4. Generate visualizations
    print("\n[4/5] Generating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'])
    
    # ROC curve
    optimal_idx = np.argmax(metrics['tpr'] - metrics['fpr'])
    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'], 
                   metrics['optimal_threshold'], optimal_idx)
    
    # Distance distribution
    plot_distance_distribution(distances, labels, metrics['optimal_threshold'])
    
    # t-SNE
    plot_tsne(embeddings, cat_ids)
    
    # 5. Generate report
    print("\n[5/5] Generating summary report...")
    
    summary = f"""
{'='*70}
COMPREHENSIVE EVALUATION SUMMARY
{'='*70}

Model: EfficientNet-B0 with Triplet Loss
Embedding Dimension: {embeddings.shape[1]}

VERIFICATION METRICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
  • Precision:  {metrics['precision']:.4f}
  • Recall:     {metrics['recall']:.4f}
  • F1-Score:   {metrics['f1_score']:.4f}
  • ROC-AUC:    {metrics['roc_auc']:.4f}

DISTANCE ANALYSIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Optimal Threshold:       {metrics['optimal_threshold']:.4f}
  • Same Cat Mean Dist:      {metrics['pos_dist_mean']:.4f} ± {metrics['pos_dist_std']:.4f}
  • Different Cat Mean Dist: {metrics['neg_dist_mean']:.4f} ± {metrics['neg_dist_std']:.4f}
  • Distance Separation:     {metrics['distance_separation']:.4f}

{'='*70}
"""
    
    print(summary)
    
    # Save summary
    with open('evaluation_summary.txt', 'w') as f:
        f.write(summary)
    
    print("✓ Evaluation summary saved as 'evaluation_summary.txt'")
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    
    return metrics
