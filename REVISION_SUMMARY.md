# Revision Summary & Implementation Details

**Note to Reviewers:** This document serves as a supplementary guide to the revisions made during the response phase for **ICASSP 2026**. It provides the formalized nomenclature table, detailed implementation settings for ablation studies, and clarified physical interpretations of the core equations.

---

## 1. Formalized Nomenclature (Address R2-3954)

To ensure rigor within the Graph Signal Processing (GSP) framework, we explicitly define the core mathematical quantities used in our **Spectral Graph Wavelet Filter Bank**.

| Symbol                    | Definition                     | Physical Meaning                                             |
| :------------------------ | :----------------------------- | :----------------------------------------------------------- |
| $X$                       | Multi-view Dataset             | Collection of $V$ views $\{X^{(v)}\}_{v=1}^V$.               |
| $\mathcal{G}^{(v)}_t$     | Dynamic Graph                  | Graph topology at epoch $t$, determined by neighbor size $k_t$. |
| **$k_t$**                 | **Time-varying Neighbor Size** | The dynamic scale parameter controlling the neighborhood expansion curriculum. |
| $L_{sym}^{(v)}$           | Normalized Laplacian           | Structural operator encoding the manifold topology.          |
| **$\Lambda^{(v)}$**       | **Graph Spectrum**             | Diagonal matrix of eigenvalues used for spectral filtering.  |
| **$P_{\theta}(\Lambda)$** | **Adaptive Filter Bank**       | The core GSP operator: $\sum_{m=1}^M \theta_m \phi_m(\Lambda)$. |
| $\phi_m$                  | Haar Wavelet Basis             | The basis functions utilized to decompose the spectrum.      |
| $\theta$                  | Learnable Coefficients         | Parameters that adaptively modulate frequency bands.         |
| $Z^{(v)}$                 | Filtered Embedding             | The view-specific representation after spectral filtering.   |
| $H$                       | Consensus Representation       | The unified latent representation used for final clustering. |

---

## 2. Ablation Study: Rigorous Baselines (Address R1-5C74)

We clarify the rigorous implementation details for the ablation variants discussed in **Section 3.3**.

### A. "w/o Multiscale" Variant

* **Definition:** In this variant, the dynamic neighborhood expansion ($k_t$) is removed. The model degenerates into a **fixed single-scale graph construction**.
* **Parameter Selection (Grid Search):** To ensure a fair comparison, we did not choose the neighbor size $k$ arbitrarily. We performed a **comprehensive grid search** to find the *optimal static setting* for each dataset.
* **Optimal Static Baseline Settings:**
  * **100leaves:** Optimal static $k = \mathbf{16}$.
  * **Handwritten:** Optimal static $k = \mathbf{51}$.
* **Result:** Even compared against these optimally tuned static baselines, our dynamic method achieves a performance gain of **+4.24%**, validating the necessity of capturing hierarchical structural dependencies.

### B. "w/o Frequency" Variant

* **Definition:** In this variant, the adaptive spectral modulation module is omitted.
* **Mathematical Degeneration:** Without learnable coefficients $\theta$, the spectral processing **mathematically degenerates into a fixed ideal low-pass filter** (approximating the prior $1 - \lambda$).
* **Result:** The performance drop of **3.88%** confirms that rigid low-pass filtering oversmooths the data, suppressing **discriminative mid-frequency patterns** (e.g., textures/boundaries) that our adaptive filter successfully captures.

---

## 3. Physical Interpretations of Equations (Address R3-4CDA)

We have added the following physical interpretations to **Section 2.5** to clarify the role of the loss functions.

### Eq. (5): Spectral Regularizer as a "Low-Pass Prior"

$$\mathcal{L}_{spectral} = \| P_{\theta}(\Lambda) - (I - \Lambda) \|_F^2$$

* **Role:** Acts as a **Low-Pass Prior** for numerical stability.
* **Mechanism:** Fully learnable filters are prone to overfitting high-frequency noise in the early stages of training. This term constrains the filter to approximate a standard heat kernel ($1-\lambda$) initially, ensuring a "safe" starting point before evolving to capture complex spectral patterns.

### Eq. (7): The Joint Optimization Objective

$$\mathcal{L}_{total} = \mathcal{L}_{rec} + \alpha \mathcal{L}_{tr} + \beta \mathcal{L}_{con} + \gamma \mathcal{L}_{spectral}$$

* **Role:** Represents a **Joint Optimization Objective**.
* **Mechanism:** It orchestrates the synergy between four distinct tasks:
  1.  **Feature Fidelity:** Reconstructing the original input ($\mathcal{L}_{rec}$).
  2.  **Spatial Smoothness:** Preserving local topology ($\mathcal{L}_{tr}$).
  3.  **View Consistency:** Aligning common semantics across views ($\mathcal{L}_{con}$).
  4.  **Spectral Stability:** Regulating the filter response ($\mathcal{L}_{spectral}$).

---

## 4. Formatting Updates (Address R2-3954)

* **Citation Style:** All citations have been standardized to the **IEEE Transaction Style (Numbered)** format, replacing the inconsistent `(Author, Year)` format.
  * *Example:* **ICMVC [27]** replaced **ICMVC (AAAI’24)**.
* **Comparisons:** We have verified the correspondence of all baseline methods (e.g., DUA-Nets, CoMSC, SDSNE) to their respective citations in the references list.

## 5. Modified Figure 1

### Fashion：

![image-20251217223045619](D:\typora_image\image-20251217223045619.png)

### Handwritten：

![image-20251217223213878](D:\typora_image\image-20251217223213878.png)





## 6. Detailed Algorithm (Pseudocode) (Address R3-4CDA)

To clarify the exact implementation logic distinguishing spatial and spectral optimization paths, we provide the detailed pseudocode of our **Multi-scale Spatial-Spectral Filtering Framework**.

---

### **Algorithm: Multi-scale Spatial-Spectral Filtering Framework**

**Input:** Multi-view dataset $\mathcal{X}$, initial neighbors $k_{int}$, step size $\Delta k$, max neighbors $k_{max}$.  
**Output:** Cluster partitions $\boldsymbol{Y}$.

**Stage 1: Frequency Fitting (Spectral Initialization)**

* Initialize neighbor count $k_t \leftarrow k_{int}$.
* **Phase 1.1: Pre-training (Dynamic Structure)**
    * **While** $k_t \le k_{max}$ **do**:
        * Iteratively update graphs $\mathcal{G}^{(v)}$ (compute $\boldsymbol{W}^{(v)}$ via Euclidean distance difference) and eigen-pairs $(\Lambda^{(v)}, \boldsymbol{U}^{(v)})$.
        * **For** $iter = 1$ to $T_{pre}$ **do**:
            * Train GAE to minimize $\mathcal{L}_{rec} + \mathcal{L}_{tr}$ (using fixed target filter).
            * Train Wavelet $\theta$ to minimize $\mathcal{L}_{sp}$ (fitting low-pass prior).
        * **End For**
        * $k_t \leftarrow \min(k_t + \Delta k, k_{max})$.
    * **End While**
* **Phase 1.2: Fine-tuning (Consistency Alignment)**
    * **For** $epoch = 1$ to $T_{fine}$ **do**:
        * Add consistency loss $\mathcal{L}_{con}$ and update GAE and Wavelet separately.
    * **End For**

**Stage 2: Joint Spatial-Spectral Optimization**
* **Substitute** target filter with learned wavelet filter $\boldsymbol{P}_{\theta}$.
* Reset neighbors $k_t \leftarrow k_{int}$.
* **Phase 2.1: Joint Pre-training (Dynamic Structure)**
    * **While** $k_t \le k_{max}$ **do**:
        * Iteratively update graphs $\mathcal{G}^{(v)}$ and eigen-pairs $(\Lambda^{(v)}, \boldsymbol{U}^{(v)})$.
        * **For** $iter = 1$ to $T_{pre}$ **do**:
            * Construct dynamic filter $\boldsymbol{P}_{\theta}^{(v)}$ and forward GAE.
            * Update **all** parameters minimizing $\mathcal{L}_{rec} + \mathcal{L}_{tr} + \beta\mathcal{L}_{sp}$.
        * **End For**
        * $k_t \leftarrow \min(k_t + \Delta k, k_{max})$.
    * **End While**
* **Phase 2.2: Joint Fine-tuning (Consistency Alignment)**
    * **For** $epoch = 1$ to $T_{fine}$ **do**:
        * Construct dynamic filter $\boldsymbol{P}_{\theta}^{(v)}$ and forward GAE.
        * Update **all** parameters minimizing $\mathcal{L}_{rec} + \mathcal{L}_{tr} + \mathcal{L}_{con} + \beta\mathcal{L}_{sp}$.
    * **End For**

**Inference (Consensus Fusion)**

* Compute consensus filter: $\overline{\boldsymbol{P}} \leftarrow \frac{1}{V}\sum_{v=1}^{V} \boldsymbol{P}_{\theta}^{(v)}$.
* Unified representation: $\boldsymbol{H} \leftarrow \overline{\boldsymbol{P}} \cdot \text{Concat}(\boldsymbol{Z}^{(1)}, \dots, \boldsymbol{Z}^{(V)})$.
* Obtain $\boldsymbol{Y}$ by performing $k$-means on $\boldsymbol{H}$.
* **Return** $\boldsymbol{Y}$