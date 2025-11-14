Generate test code to verify that FR-RoPE is rotation-invariant while standard RoPE is not.

Create a test script that:
1. Generates synthetic 2D spatial data or loads a small MERFISH sample
2. Applies both standard RoPE and FR-RoPE to the original coordinates
3. Rotates the coordinates by several angles (e.g., 30°, 90°, 180°)
4. Recomputes embeddings on rotated coordinates
5. Measures and reports the difference:
   - Standard RoPE: embeddings should change significantly with rotation
   - FR-RoPE: embeddings should remain nearly identical (up to numerical precision)
6. Visualizes the embeddings or prints numerical comparison

The test should clearly demonstrate the rotation invariance property of FR-RoPE.
