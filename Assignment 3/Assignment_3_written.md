# Assignment Answers

## **1. (a)**

### **i. Momentum in Adam Optimization**

The goal of training a machine learning model is to minimize the loss function. Gradient descent moves downhill to find the lowest point but can get stuck in small bumps (local minima) or zigzag inefficiently. Momentum in Adam optimization uses past gradient information to smooth the path, reducing zigzagging and helping the model reach the optimal solution more efficiently. It’s like using "momentum" to roll smoothly over small rocks while going downhill.

---

### **ii. Which parameters get larger updates in Adam?**

In Adam, parameters with smaller past gradients (\(v_t\), the moving average of squared gradients) will have larger updates. This is because the update is divided by \(\sqrt{v_t}\), so when \(v_t\) is small, the denominator is small, resulting in larger updates.

#### **Why is this helpful for learning?**

This mechanism is helpful for the following reasons:
- **Focus on parameters with smaller gradients**: By giving larger updates to parameters with smaller gradients, Adam ensures that these parameters still make progress, even when their gradients are tiny.
- **Prevent over-updating for large gradients**: For parameters with large gradients, smaller updates help avoid overshooting the optimal solution and keep learning stable.

#### **Intuition with an example**

Imagine navigating a bumpy 3D landscape (loss function). In flatter regions (small gradients), Adam takes larger steps to ensure progress, while in steep regions (large gradients), it takes smaller steps to avoid overshooting. This allows Adam to efficiently and smoothly reach the global minimum, avoiding both stagnation in flat areas and instability in steep areas.

---

## **1. (b) Dropout: Answer**

---

### **i. What must \(\gamma\) equal in terms of \(p_{\text{drop}}\)?**

Dropout randomly deactivates units in the hidden layer \(h\) with probability \(p_{\text{drop}}\) and scales the output \(h_{\text{drop}}\) to ensure that its expected value matches the original \(h\). The relationship is given by:

\[
h_{\text{drop}, i} = \gamma \, d_i \, h_i
\]

Here:
- \(d_i\): A mask variable that is \(1\) with probability \(1 - p_{\text{drop}}\) and \(0\) with probability \(p_{\text{drop}}\).

To compute the expected value:

\[
E[h_{\text{drop}, i}] = \gamma \, h_i \, E[d_i]
\]

Since \(E[d_i] = 1 - p_{\text{drop}}\), we have:

\[
E[h_{\text{drop}, i}] = \gamma \, h_i \, (1 - p_{\text{drop}})
\]

To ensure that \(E[h_{\text{drop}, i}] = h_i\), solve for \(\gamma\):

\[
\gamma \cdot (1 - p_{\text{drop}}) = 1 \implies \gamma = \frac{1}{1 - p_{\text{drop}}}
\]

**Conclusion**:  
The scaling factor \(\gamma\) is:

\[
\gamma = \frac{1}{1 - p_{\text{drop}}}
\]

---

### **ii. Why should dropout be applied during training but NOT during evaluation?**

#### **Why apply dropout during training?**  
Dropout acts as a regularization technique to improve generalization by preventing overfitting. Specifically:
- By randomly deactivating units in the hidden layer, the model cannot rely on specific units and must distribute learning across all units.
- Deactivating different units for each minibatch forces the model to learn more robust representations, improving its ability to generalize to unseen data.

#### **Why NOT apply dropout during evaluation?**  
During evaluation, all units should remain active to fully utilize the model's learned capacity. Dropout is not applied for the following reasons:

1. **Consistency**:  
   - Applying dropout would introduce randomness, making predictions unreliable.  

2. **Scaling**:  
   - At evaluation, the scaling factor \(\gamma = \frac{1}{1 - p_{\text{drop}}}\) is applied to ensure the output matches the expected value during training. This makes dropout unnecessary during evaluation.

---

**Summary**:  
- **Training**: Dropout prevents overfitting by deactivating units randomly and encouraging the model to generalize.  
- **Evaluation**: Dropout is not applied to ensure stable and consistent predictions with the full model.

---

## **2. (a) Dependency Parsing**

### **Parsing the sentence: "I parsed this sentence correctly."**

| Stack                          | Buffer                                 | New Dependency      | Transition            |
| ------------------------------ | -------------------------------------- | ------------------- | --------------------- |
| [ROOT]                         | [I, parsed, this, sentence, correctly] |                     | Initial Configuration |
| [ROOT, I]                      | [parsed, this, sentence, correctly]    |                     | SHIFT                 |
| [ROOT, I, parsed]              | [this, sentence, correctly]            |                     | SHIFT                 |
| [ROOT, parsed]                 | [this, sentence, correctly]            | parsed -> I         | LEFT-ARC              |
| [ROOT, parsed, this]           | [sentence, correctly]                  |                     | SHIFT                 |
| [ROOT, parsed, this, sentence] | [correctly]                            |                     | SHIFT                 |
| [ROOT, parsed, sentence]       | [correctly]                            | sentence -> this    | LEFT-ARC              |
| [ROOT, parsed]                 | [correctly]                            | parsed -> sentence  | RIGHT-ARC             |
| [ROOT, parsed, correctly]      | []                                     |                     | SHIFT                 |
| [ROOT, parsed]                 | []                                     | parsed -> correctly | RIGHT-ARC             |
| [ROOT]                         | []                                     | ROOT -> parsed      | RIGHT-ARC             |

---

## **2. (b) Steps for Dependency Parsing**

The dependency parsing of a sentence with \(n\) words requires **\(2n + 1\) steps**.

This is broken down as follows:
- **Initialization**: 1 step  
- **SHIFT operations**: \(n\) steps (one for each word moved from Buffer to Stack)  
- **LEFT-ARC/RIGHT-ARC operations**: \(n\) steps (one for each word removed from Stack)

**Total**: \(1 + n + n = 2n + 1\) steps

---

## **2. (f) Error Types in Dependency Parsing**

### **i. Sentence**  
**I disembarked and was heading to a wedding fearing my death.**  

- **Error Type**: Verb Phrase Attachment Error  
- **Incorrect Dependency**: wedding -> fearing  
- **Correct Dependency**: disembarked -> fearing  

---

### **ii. Sentence**  
**It makes me want to rush out and rescue people from dilemmas of their own making.**  

- **Error Type**: Coordination Attachment Error  
- **Incorrect Dependency**: rescue -> and  
- **Correct Dependency**: rescue -> rush  

---

### **iii. Sentence**  
**It is on loan from a guy named Joe O'Neill in Midland, Texas.**  

- **Error Type**: Prepositional Phrase Attachment Error  
- **Incorrect Dependency**: named -> Midland  
- **Correct Dependency**: O'Neill -> Midland  

---

### **iv. Sentence**  
**Brian has been one of the most crucial elements to the success of Mozilla software.**  

- **Error Type**: Modifier Attachment Error  
- **Incorrect Dependency**: elements -> most, elements -> the  
- **Correct Dependency**: crucial -> most, most -> the  
