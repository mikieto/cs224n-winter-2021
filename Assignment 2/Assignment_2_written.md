**Assignment 2: Written Solutions**

**(a) (3 points) Show that the naive-softmax loss given in Equation (2) is the same as the cross-entropy loss between y and ŷ; i.e., show that**

$\sum_{w \in \text{Vocab}} y_w \log(\hat{y}_w) = - \log(\hat{y}_o)$

**Answer:**

The true distribution `y` is a "one-hot" vector. This means that only the position corresponding to the actual outside word has a value of 1, and all other positions have a value of 0. So, the cross-entropy formula becomes:

$\sum_{w \in \text{Vocab}} y_w \log(\hat{y}_w) =  y_o \log(\hat{y}_o) = 1 \cdot \log(\hat{y}_o) = \log(\hat{y}_o)$

Here, $\hat{y}_o$ represents the predicted probability $P(O=o|C=c)$ of the outside word `o` given the center word `c`. In Equation (2), the loss function is the negative log of this probability, so

$ - \log(P(O=o|C=c)) = -\log(\hat{y}_o)$

which is the same as the cross-entropy.

**Key Points:**

*   A one-hot vector only has one "1" and the rest are "0".
*   The predicted probability $\hat{y}_o$ is the same as $P(O=o|C=c)$.

**(b) (5 points) Compute the partial derivative of Jnaive-softmax(vc, o, U) with respect to vc. Please write your answer in terms of y, ŷ, and U. Note that in this course, we expect your final answers to follow the shape convention. This means that the partial derivative of any function f(x) with respect to x should have the same shape as x. For this subpart, please present your answer in vectorized form. In particular, you may not refer to specific elements of y, ŷ, and U in your final answer (such as y1, y2, ...).**

**Answer:**

$\frac{\partial J_{\text{naive-softmax}}}{\partial \mathbf{v}_c} = U(\hat{\mathbf{y}} - \mathbf{y})$

**Explanation:**
This formula represents how much the loss function changes with a small change in the center word vector `vc`.

**Key Points:**
* Use the chain rule to break down the derivative.
*  Express the final result using vectors `y`, `ŷ`, and the matrix `U`.
*  Follow the shape convention, not referring to specific elements.

**(c) (5 points) Compute the partial derivatives of Jnaive-softmax(vc, o, U) with respect to each of the ‘outside' word vectors, uw's. There will be two cases: when w = o, the true 'outside' word vector, and w ≠ o, for all other words. Please write your answer in terms of y, ŷ, and vc. In this subpart, you may use specific elements within these terms as well, such as (y1, y2, ...).**

**Answer:**

*   **Case 1: `w = o` (when `uw` is the correct outside word vector):**

    $\frac{\partial J_{\text{naive-softmax}}}{\partial \mathbf{u}_o} = (\hat{y}_o - 1) \mathbf{v}_c$

*   **Case 2: `w ≠ o` (when `uw` is *not* the correct outside word vector):**

    $\frac{\partial J_{\text{naive-softmax}}}{\partial \mathbf{u}_w} = \hat{y}_w \mathbf{v}_c$

**Explanation:**
This formula tells us how to adjust the outside word vectors to improve the loss. We have two different cases for the correct and incorrect word vectors.

**Key Points:**

*   Consider two cases for the outside word vector (`uw`).
*   Show the derivative using `y`, `ŷ`, and `vc` and can use specific elements.

**(d) (1 point) Compute the partial derivative of Jnaive-softmax(vc, o, U) with respect to U. Please write your answer in terms of $\frac{\partial J(v_c, o, U)}{\partial u_1}$, $\frac{\partial J(v_c, o, U)}{\partial u_2}$ , ... $\frac{\partial J(v_c, o, U)}{\partial u_{V_{ocab}}}$. The solution should be one or two lines long.**

**Answer:**

$\frac{\partial J_{\text{naive-softmax}}}{\partial U} = \left[ \frac{\partial J_{\text{naive-softmax}}}{\partial \mathbf{u}_1}, \frac{\partial J_{\text{naive-softmax}}}{\partial \mathbf{u}_2}, ..., \frac{\partial J_{\text{naive-softmax}}}{\partial \mathbf{u}_{|V|}} \right] = \mathbf{v}_c (\hat{\mathbf{y}} - \mathbf{y})^\mathrm{T}$

**Explanation:**
This formula shows the partial derivative of the loss with respect to all the outside vectors at once, represented in matrix U.

**Key Points:**

*   Matrix U consists of all outside word vectors.
*   The result should be expressed as a collection of individual derivatives.

**(e) (3 Points) The sigmoid function is given by Equation 4:**

$\sigma(x) = \frac{1}{1+ e^{-x}} = \frac{e^x}{e^x + 1}$

**Please compute the derivative of σ(x) with respect to x, where x is a scalar. Hint: you may want to write your answer in terms of σ(x).**

**Answer:**

$\frac{d\sigma(x)}{dx} = \sigma(x)(1 - \sigma(x))$

**Explanation:**
This formula represents how much the sigmoid function changes with a small change in x.

**Key Points:**

*   Apply the chain rule of differentiation.
* Express the derivative using the sigmoid function itself.

**(f) (4 points) Now we shall consider the Negative Sampling loss, which is an alternative to the Naive Softmax loss. Assume that K negative samples (words) are drawn from the vocabulary. For simplicity of notation we shall refer to them as w1, w2,...,wk and their outside vectors as u1,..., uk. For this question, assume that the K negative samples are distinct. In other words, i ≠ j implies wi ≠ wj for i, j∈ {1,...,K}. Note that o ∉ {w1,...,wk}. For a center word c and an outside word o, the negative sampling loss function is given by:**

$J_{neg-sample}(v_c, o, U) = − log(σ(u_o^T v_c)) − \sum_{k=1}^{K} log(σ(−u_{w_k}^T v_c))$

**Please repeat parts (b) and (c), computing the partial derivatives of Jneg-sample with respect to vc, with respect to uo, and with respect to a negative sample uk. Please write your answers in terms of the vectors uo, vc, and uk, where k ∈ [1, K]. After you've done this, describe with one sentence why this loss function is much more efficient to compute than the naive-softmax loss. Note, you should be able to use your solution to part (e) to help compute the necessary gradients here.**

**Answer:**

*   **Partial derivative with respect to the center word vector `vc`:**

    $\frac{\partial J_{\text{neg-sample}}}{\partial \mathbf{v}_c} = - (1 - \sigma(\mathbf{u}_o^\mathrm{T} \mathbf{v}_c)) \mathbf{u}_o + \sum_{k=1}^{K} (1 - \sigma(-\mathbf{u}_{w_k}^\mathrm{T} \mathbf{v}_c)) \mathbf{u}_{w_k}$

*   **Partial derivative with respect to the outside word vector `uo` (the true outside word):**

    $\frac{\partial J_{\text{neg-sample}}}{\partial \mathbf{u}_o} = (\sigma(\mathbf{u}_o^\mathrm{T} \mathbf{v}_c) - 1) \mathbf{v}_c$

*   **Partial derivative with respect to a negative sample word vector `uk`:**

    $\frac{\partial J_{\text{neg-sample}}}{\partial \mathbf{u}_{w_k}} = (1 - \sigma(-\mathbf{u}_{w_k}^\mathrm{T} \mathbf{v}_c)) \mathbf{v}_c$

*   **Efficiency:**

    Negative Sampling Loss is much more efficient than Naive Softmax Loss because it only calculates the probabilities for the true outside word and a small number of negative examples, instead of all the words in the vocabulary.

**Explanation:**
These formulas show how to adjust the word vectors based on the negative sampling loss.

**Key Points:**

*   Use the derivative of the sigmoid function.
* Consider the three cases: with respect to center word, correct outside word, and negative samples.
* Negative Sampling Loss is computationally efficient as it only deals with a subset of all the words, not all the words in the vocabulary as in Naive Softmax Loss

**(g) (2 point) Now we will repeat the previous exercise, but without the assumption that the K sampled words are distinct. Assume that K negative samples (words) are drawn from the vocabulary. For simplicity of notation we shall refer to them as w1, w2,...,wk and their outside vectors as u1,...,UK. In this question, you may not assume that the words are distinct. In other words, wi = wj may be true when i ≠ j is true. Note that o ∉ {w1,...,wk}. For a center word c and an outside word o, the negative sampling loss function is given by:**
$J_{neg-sample}(v_c, o, U) = − log(σ(u_o^T v_c)) − \sum_{k=1}^{K} log(σ(−u_{w_k}^T v_c))$

**Compute the partial derivative of Jneg-sample with respect to a negative sample uk. Please write your answers in terms of the vectors vc and uk, where k ∈ [1, K]. Hint: break up the sum in the loss function into two sums: a sum over all sampled words equal to uk and a sum over all sampled words not equal to uk.**

**Answer:**

$\frac{\partial J_{\text{neg-sample}}}{\partial \mathbf{u}_{w_k}} = |S|(1 - \sigma(-\mathbf{u}_{w_k}^\mathrm{T} \mathbf{v}_c)) \mathbf{v}_c$

Here, $|S|$ is the number of times the negative sample $\mathbf{u}_{w_k}$ appears in the set of K negative samples.

**Explanation:**
This formula takes into account the case when negative samples can be repeated.

**Key Points:**

*   Consider the case where negative samples can be the same.
*   The derivative includes the number of times the negative sample appears.

**(h) (3 points) Suppose the center word is c = wt and the context window is [Wt-m, ..., Wt-1, Wt, Wt+1, ..., Wt+m], where m is the context window size. Recall that for the skip-gram version of word2vec, the total loss for the context window is:**
$J_{skip-gram}(v_c, w_{t-m},...,w_{t+m}, U) = \sum_{-m<j<m, j\ne 0} J(v_c, w_{t+j}, U)$

**Here, J(vc, Wt+j, U) represents an arbitrary loss term for the center word c = wt and outside word Wt+j. J(vc, Wt+j, U) could be Jnaive-softmax(vc, Wt+j, U) or Jneg-sample(vc, Wt+j, U), depending on your implementation.
Write down three partial derivatives:**

**(i) dJskip-gram(Vc, Wt-m,... Wt+m,U)/dU
(ii) Jskip-gram(Vc, Wt-m,... Wt+m, U)/dvc
(iii) dJskip-gram(Vc, Wt-m,... Wt+m,U)/dvw when w ≠ c
Write your answers in terms of dJ(vc, Wt+j,U)/JU and dJ(vc,Wt+j,U)/dvc. This is very simple – each solution should be one line.**

**Answer:**

(i) $\frac{\partial J_{\text{skip-gram}}(v_c, w_{t-m}, ..., w_{t+m}, U)}{\partial U} = \sum_{-m \le j \le m, j \ne 0} \frac{\partial J(\mathbf{v}_{w_t}, w_{t+j}, U)}{\partial U}$

(ii) $\frac{\partial J_{\text{skip-gram}}(v_c, w_{t-m}, ..., w_{t+m}, U)}{\partial \mathbf{v}_c} = \sum_{-m \le j \le m, j \ne 0} \frac{\partial J(\mathbf{v}_{w_t}, w_{t+j}, U)}{\partial \mathbf{v}_{w_t}}$

(iii) $\frac{\partial J_{\text{skip-gram}}(v_c, w_{t-m}, ..., w_{t+m}, U)}{\partial \mathbf{v}_w} = \sum_{-m \le j \le m, j \ne 0} \frac{\partial J(\mathbf{v}_{w_t}, w_{t+j}, U)}{\partial \mathbf{v}_w}$

**Explanation:**
These formulas show the partial derivatives of the total loss for a context window with respect to different parameters.

**Key Points:**

*   The total loss is the sum of individual losses for all outside words.
*   The derivatives are just the sum of derivatives of individual loss terms.
*    The partial derivatives are calculated with respect to the model parameters U and the center word vector vc for all context words and the outside word vector vw where w is not the center word c.
