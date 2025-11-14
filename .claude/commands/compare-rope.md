Run a comparison experiment between standard RoPE and FR-RoPE on the same Transformer model.

Create or modify training code to:
1. Train two identical Transformer models:
   - Model A: uses standard axial RoPE
   - Model B: uses FR-RoPE (frame-rotation-invariant)
2. Use the same MERFISH dataset with rotation augmentation
3. Track and compare:
   - Training loss and accuracy
   - Validation performance
   - Robustness to rotated test data
4. Generate comparison plots or tables showing:
   - Learning curves
   - Performance on rotated vs non-rotated test sets
   - Embedding visualization (if applicable)

The goal is to empirically validate whether FR-RoPE provides better rotation robustness in a real Transformer model on MERFISH data.
