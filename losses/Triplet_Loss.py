import torch
import torch.nn as nn

# --- Example Usage ---
embedding_dim = 128
margin = 0.5
batch_size = 4 # Example with a batch of triplets

# Assume these tensors are outputs from your model branches (e.g., image and text encoders)
# Shape: (batch_size, embedding_dim)
anchor_embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)
positive_embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)
negative_embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)

# --- Using Euclidean Distance (default, p=2) ---
# Instantiate the loss function
# margin: The margin value
# p: The norm degree for pairwise distance. Default: 2 (Euclidean)
# reduction: 'mean' (average loss over batch), 'sum', or 'none'
triplet_loss_fn_euclidean = nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')

# Calculate the loss
loss_euclidean = triplet_loss_fn_euclidean(anchor_embeddings, positive_embeddings, negative_embeddings)

print(f"\nPyTorch Example (Euclidean):")
print(f"Triplet Loss (Euclidean, mean reduction): {loss_euclidean.item():.4f}")

# --- Using Cosine Distance ---
# For Cosine distance, you typically maximize similarity. Triplet loss uses distance.
# Cosine Distance = 1 - Cosine Similarity
# PyTorch doesn't have a direct TripletMarginLoss for Cosine *distance* easily,
# but you can implement it or use TripletMarginWithDistanceLoss with CosineSimilarity.

# Option A: Manual calculation with cosine distance
def cosine_distance(t1, t2, eps=1e-8):
    return 1 - nn.functional.cosine_similarity(t1, t2, dim=-1, eps=eps)

d_ap_cos = cosine_distance(anchor_embeddings, positive_embeddings)
d_an_cos = cosine_distance(anchor_embeddings, negative_embeddings)
loss_cos_manual = torch.mean(torch.relu(d_ap_cos - d_an_cos + margin)) # relu is max(0, x)

print(f"\nPyTorch Example (Cosine - Manual):")
print(f"Triplet Loss (Cosine, mean reduction): {loss_cos_manual.item():.4f}")

# Option B: Using TripletMarginWithDistanceLoss (more flexible)
triplet_loss_fn_cosine = nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: 1 - nn.functional.cosine_similarity(x, y),
    margin=margin,
    reduction='mean'
)
loss_cosine_builtin = triplet_loss_fn_cosine(anchor_embeddings, positive_embeddings, negative_embeddings)

print(f"\nPyTorch Example (Cosine - Built-in Wrapper):")
print(f"Triplet Loss (Cosine, mean reduction): {loss_cosine_builtin.item():.4f}")


# Backpropagation example (loss_euclidean used here)
# loss_euclidean.backward()
# optimizer.step() # Assuming an optimizer is defined
# print("\nGradients computed (example):")
# print("Anchor grad sample:", anchor_embeddings.grad[0, :5]) # Print first 5 grad elements of first batch item
