Assignment 4

### **1. (g) Answer**

1. **Effect of the mask**:
   - `enc_masks` sets the attention scores for `PAD` tokens in the source sentence to a very small value (\(-\infty\)). This ensures that, after applying the softmax function, the attention weights for the `PAD` tokens become close to 0 and are effectively ignored.
   - As a result, the attention mechanism focuses only on meaningful tokens, allowing the model to accurately extract relevant contextual information.
   - Since the softmax function normalizes the scores, setting the `PAD` token scores to \(-\infty\) prevents them from influencing the distribution and ensures that attention is allocated exclusively to meaningful tokens.

2. **Necessity of the mask**:
   - `PAD` tokens are added to pad sentences to a uniform length but do not contribute to the actual translation or contextual understanding. If they are not ignored, the attention mechanism would incorrectly allocate weights to them, leading to reduced translation quality.
   - By setting the scores to \(-\infty\), the softmax ensures that the attention weights for `PAD` tokens are 0, completely removing their influence. This approach is essential to maintain the accuracy and integrity of the attention distribution.
