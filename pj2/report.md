# Project 2 Report

---

**Name** : XXX |   **Student ID** : XXX  |  **Due** **Date** : 2025/06/10

**GitHub Code Link** : [https://github.com/tyyyyy333/Fdu_DS_DL24spring_collection/blob/main/pj2/]

**Model Weights + Tensorboard + Dataset Link** : [https://drive.google.com/drive/folders/1hX6oN6LV9J3zQWyNWiULx6eAfFiNusu-?usp=sharing]

> for the visualization, we use the [Tensorboard](https://www.tensorflow.org/tensorboard) to visualize the training process. But the tensorboard is not included in the project, you can use the following command to visualize the training process:
>
> ```bash
> tensorboard --logdir=runs
> ```
>
> Make sure you have installed the tensorboard before running the command.
> `pip install tensorboard`
> The logger files are uploaded to the Netdisk, you can download them and put them into the `runs` folder thus running the bash command above.

---

## Part 1: CIFAR-10 Classification

### 1.1 Dataset and Preprocessing

We use the **CIFAR-10** dataset, which contains 60,000 images (32×32×3) evenly distributed over 10 classes. The training-validation split was performed with a validation ratio of 0.1.

* Data augmentation: `RandomHorizontalFlip`, `RandomCrop(32,padding=4)`, `RandomErasing`
* Normalization: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
* We also experimented with **Mixup** and **CutMix** data mixing strategies to improve generalization.

### 1.2 Network Architecture

We propose a custom architecture named  **CANet** , the CANet (Convolution-and-Attention Network) is designed to effectively extract multi-scale spatial and semantic features through a carefully structured stack of attention-enhanced residual blocks (AttResBlock). Each AttResBlock integrates both channel-wise and spatial attention mechanisms, enabling the network to adaptively focus on the most informative features across dimensions.

The AttResBlock is the core component of our model, designed to enhance feature representation by combining dual convolutional paths with residual connections and attention mechanisms.

The module consists of:

- Two main paths:
  Each path has two convolutional layers with ReLU and BatchNorm.
- Path 1 passes through a Spatial Attention (SA) module.
- Path 2 passes through a Multi-kernel Channel Attention (SE) module.
- Shared unit transformation:
  Outputs from both attention modules are processed by a shared unit_transform block to produce attention weights.
- Residual connection:
  The input is downsampled, ensuring  **gradient flow and feature preservation** .
- The final output is the **summation** of the two reweighted main paths and the residual path, followed by ReLU activation.

Motivations:

* The dual-branch structure ensures that both attention mechanisms influence the feature transformation in a  **controlled and disentangled way** , which avoids mutual interference.
* By merging two refined feature maps (channel & spatial aware), the model ensures graphic semantic consistency from the branches by restricting with the shared block and achieves **better generalization** on complex image patterns, as well as bettering the training process.
* These branches allow the network to attend to **"which"** (channel semantics) and **"where"** (spatial location) information separately.This separation encourages  **diverse and complementary feature extraction** .

<img src="cifar10\figs\scratch.png" width="800" height="800">

- u1, u2 : attention weights from `unit_transform(SA(out1))` and `unit_transform(SA(out2))`
- residual: original input (optionally downsampled)

An architecture of our network is shown in `tensorboard` in the form of model summary.

Here is an scratch of the model:

> Input
> ↓
> AttResBlock × 6
> ↓
> AvgPool → BatchNorm → Flatten
> ↓
> FC1 → ReLU → FC2 → Output

- We abandoned dropout as a regularization strategy, partly because we simplified the linear layer classification to reduce the parameters, and partly because we observed that dropout had no significant effect on performance. In the same way, we abandoned the enhancement measures such as color enhancement and random rotation when the data was augmented, because it was observed that these measures did not improve the generalization performance.
- In terms of training, we choose to use the AdamW optimizer for fast convergence in the early stage of training and the SGD optimizer to try optimization at the end of training, which simplifies the adjustment of the optimization strategy and can avoid the problem of unstable and slow optimization caused by directly using the SGD optimizer.

### 1.3 Optimization Strategies

* Optimizer: `AdamW when epoch <= 0.8 * total epochs`, `SGD with momentum=0.9 when epoch > 0.8 * total epochs`
* Learning Rate Scheduler: `ExponentialLR`, `γ=0.96`, `init_lr=0.001`
* Loss function: `CrossEntropy with Label Smoothing (0.1)`
* Regularization:  `Weight Decay= 1e-3`, `Data Augmentation`
* Training Epochs: `70`
* Batch Size: `256`
* Device: `CUDA`


> - Different activation functions were considered during model design. While ReLU was used throughout the majority of the network due to its computational simplicity and strong empirical performance in deep convolutional architectures, Sigmoid activation was specifically employed in the attention mechanism(SA/SE/unit_transform)) This choice was intentional: Sigmoid maps inputs to a [0, 1] range, making it particularly suitable for generating soft attention masks that act as multiplicative gates.
> - Although other activations such as Leaky ReLU or GELU were not ultimately adopted, their use was considered in early design discussions. ReLU was retained for its **effective** gradient propagation and **sparse** activation, both of which contribute to stable and efficient training.

> - Different loss functions and regularization techniques were carefully considered. Although we ultimately adopted Cross-Entropy Loss with Label Smoothing (ε = 0.1), this choice was motivated by its proven effectiveness in classification tasks. Cross-Entropy remains the standard for multi-class classification due to its probabilistic interpretability and gradient stability. To further enhance its generalization ability, label smoothing was introduced, which prevents the model from becoming overconfident in predictions and encourages better calibration.
> - As for regularization, we applied Weight Decay (1e-3) to penalize large weights and reduce overfitting, and adopted Data Augmentation techniques such as random crop and horizontal flip to enrich the training distribution. These methods, though simple, significantly contributed to performance robustness and generalization.

> - The training process further incorporated an AdamW optimizer during the early epochs (epoch ≤ 0.8 * total_epochs), switching to SGD with momentum (0.9) later to stabilize convergence. An ExponentialLR scheduler with decay rate γ = 0.96 controlled the learning rate across 70 epochs, with a batch size of 256. All models were trained and evaluated on CUDA-enabled GPUs to ensure efficiency.


---

### 1.4 Training Results

* ALL the training results, model definition and visualization of training process will be displayed in **tensorboard**.
  In the cifar10 folder, run the following command line:
  `tensorboard --logdir=runs/[CANet or else]`
  Note : model definition written by `torchinfo.summary` will be displayed in text field.

  The graph displayed as followed in *1.5* will be a glimpse of visualization.

---

For comparison, we use the **same training parameters** to train each valid model and give the corresponding test set results to illustrate the efficiency and simplicity of our model. We provide an enlarged and scaled down version of the original CANet for comparison, with ratios of 2x, 0.5x, and 0.25x, respectively, and we can see the model definition in the `model.py` file, and for models such as resnet, we adjust the classification layer and shallow layer accordingly to suit the resolution of the CIFAR dataset

|           Model           |    Test Accuracy    |     Total Parameters     |
| :-----------------------: | :------------------: | :----------------------: |
|        Shufflenet        |        62.66%        |         352,042         |
| ***CANet_tiny*** | ***88.66%*** |  ***396,155***  |
| ***CANet_light*** | ***91.56%*** | ***1,576,244*** |
|        MobileNetV2        |        84.14%        |        2,236,682        |
|      ***CANet* **      | ***93.50%*** | ***6,289,574*** |
|         Resnet18         |        93.00%        |        11,173,962        |
|         Resnet34         |        93.56%        |        21,282,122        |
|  ***CANet_pro***  | ***93.56%*** | ***25,128,842*** |
|           VGG11           |        85.75%        |       128,807,306       |

Here we can see that our network structure can achieve a high accuracy of the test dataset with a low number of parameters, and can still maintain a certain efficiency when the number of parameters decreases a lot, and the network is easy to converge, and can be quickly optimized to a high accuracy through the AdamW optimizer.

---

### Training Curves

> - Here comes the learning rate landscape
>
> <img src="cifar10\figs\lr.png" width="700" height="500">

> - Here comes the loss landscape
>
> <img src="cifar10\figs\loss_epoch_train.png" width="700" height="500">
> <img src="cifar10\figs\loss_epoch_val.png" width="700" height="500">

> - Here comes the accuracy landscape
>
> <img src="cifar10\figs\acc_val.png" width="700" height="500">

---

### 1.5 Visualizations and Discussion

#### 1.5.1 Filter Visualization

> <img src="cifar10\figs\conv.png" width="1000" height="500">
>
> We can see that the convolutional kernel parameters are mostly distributed around zero due to the BN layer.

#### 1.5.2 Attention Module Activation

> - SELayer:
>
> <img src="cifar10\figs\se.png" width="1000" height="500">

> - SALayer:
>
> <img src="cifar10\figs\sa.png" width="1000" height="500">

From the visualization results, it can be observed that the weight distribution of the SA module is relatively smooth, mainly concentrated in the range of -0.1 to 0.2, and there are obvious multi-spike fluctuations in the distribution range. This suggests that the model learns a wide range of responsiveness to spatial regions on this branch, corresponding to the design goal of spatial attention, i.e., to dynamically model "where is more important in the image" positional features through convolutional dynamics. In addition, the weight distribution of different channels fluctuates significantly, indicating that the SA module has strong spatial sensitivity during the training process and can better capture local structure information.

In contrast, the SE module exhibits a more sparse weight distribution, with prominent peaks in specific channels while the majority of weights are near zero. This selective activation reflects the intended behavior of the channel attention mechanism: to emphasize critical channels while suppressing irrelevant ones. This property improves the model’s capacity for feature selection and generalization.

In summary, both SA and SE branches demonstrate weight patterns consistent with their functional objectives, validating the effectiveness of incorporating attention mechanisms along both spatial and channel dimensions.

#### 1.5.3 United Transformation

> -unit_transform
>
> <img src="cifar10\figs\unit.png" width="1000" height="500">

From the visualization results, it can be observed that the weight distribution of the unit_transform block is relatively compact and symmetric, primarily centered around zero, with the majority of values falling within the range of -0.15 to 0.15. This indicates that the module maintains stable parameter magnitudes during training. The smooth, bell-shaped pattern across training steps suggests consistent optimization behavior, without abrupt oscillations or instability.

Notably, the layered ridge plot reveals minimal outliers and a dense central mass of weights and has a faint doublet shape, implying that the module enforces a form of low-rank transformation. This is in line with its assumed role as a projection or transformation unit—potentially bridging different stages of the network by aligning feature dimensions or refining intermediate representations. Compared to attention modules, the unit_transform block exhibits neither sharp activations nor sparse dominance, but rather a balanced, continuous adaptation pattern.

In summary, the unit_transform module exhibits a well-regularized and evenly distributed weight structure, reflecting its function as a stable conduit for inter-block communication. The absence of extreme activations or sparsity further suggests that this component operates as a smooth transformation layer, complementing the selective behavior of the attention mechanisms in the surrounding architecture.

---

## Part 2: Batch Normalization Analysis

### 2.1 BN vs. No BN: Performance Comparison

- All the landscape will be exposed in the `tensorborad`,
- Run the tensorboard by `tensorboard --logdir=runs/comparison` in `codes/VGG_BatchNorm` folder.

We constructed two versions of VGG_A:

* **With BatchNorm** : default version (VGG_A_BatchNorm)
* **Without BatchNorm** : we removed all `nn.BatchNorm2d` layers (VGG_A)

As an example, we choose learning rate 0.001 and trained both versions for 50 epochs.

| Model      | Best Val Accuracy |
| ---------- | ----------------- |
| With BN    | **84.29%**  |
| Without BN | 75.81%            |

> - loss curve of VGG_A_BatchNorm:
>
> <img src="codes\VGG_BatchNorm\figs\loss_epoch_val_of_vgg_bn_0.001.png" width="1000" height="500">

> - loss curve of VGG_A:
>
> <img src="codes\VGG_BatchNorm\figs\loss_epoch_val_of_vgg_nobn_0.001.png" width="1000" height="500">

 We can see that the accuracy curve of the model with BN layer on the validation set is smoother, the convergence speed is faster, and the final accuracy is higher under the same epoch, indicating that the BN layer significantly accelerates the training process and verifies the effect of stable training

---

### 2.2 Effect of BN on Optimization Landscape

At this section, we trained the model under multiple learning rates [0.001, 0.002, 0.0005, 0.0001] and plotted the **loss variation** at each step.

* We recorded the maximum and minimum loss across settings and visualized the region between curves.

> <img src="codes\VGG_BatchNorm\figs\loss_range.png" width="1000" height="500">
>
> In the case of BN, the area of the Loss curve is significantly narrowed, indicating that it is more stable to gradient change and the optimized surface is smoother. In the case of non-bn, the loss fluctuates greatly, and there is a loss explosion in the late training stage due to the Adam series optimizer

- For more comparison, we plot the gradient range of the final layer of each model under learning rate 0.0001 to see the effect of stable training of BatchNorm.

> <img src="codes\VGG_BatchNorm\figs\grad_range.png" width="1000" height="500">
> Here we can see that the model with the BN layer converges significantly faster between the gradient maximum and minimum values of the last layer, and the gradient range is smaller than that of the model without the BN layer in the later training stage which indicates that the BN layer can effectively stabilize the training process and improve the optimization performance of the model.

---

### 2.3 Interpretation

The experimental results support that:

* BN **accelerates convergence** by stabilizing the input distribution of layers；
* BN leads to  **smoother loss landscapes** , improving the gradient predictability；

---

## Conclution

In the first part of this project, we present CANet, a compact yet expressive convolutional neural network designed for image classification on the CIFAR-10 dataset. By incorporating dual-branch residual blocks, with channel attention (SE) and spatial attention (SA) modules applied independently in each branch, CANet achieves a refined feature representation that efficiently captures both semantic and spatial dependencies.

Through comprehensive experiments, we show that CANet achieves competitive classification accuracy compared with larger architectures such as ResNet and VGG, despite having significantly fewer parameters. The Light and Pro variants of CANet offer clear trade-offs between model complexity and accuracy, making the architecture flexible and scalable for diverse application scenarios. The model also converges faster and demonstrates better generalization with fewer training epochs.

In the second part of this project, we conduct a targeted investigation into the effect of Batch Normalization (BN) by comparing two variants of the VGG_A model: one with BN layers inserted after each convolutional layer, and one without.

Our experiments reveal that adding Batch Normalization significantly improves both the training stability and generalization performance of the network. The BN-augmented model not only converges faster and reaches higher final accuracy, but also exhibits smoother gradient flow and reduced internal covariate shift, as observed through per-step gradient range visualization and loss curves.

By analyzing metrics such as step-wise loss fluctuations, validation accuracy progression, and gradient magnitude ranges, we demonstrate that BN effectively regularizes the network, making it less sensitive to initialization and learning rate selection. This controlled setup isolates the influence of BN and confirms its critical role in enabling deeper architectures to train efficiently and generalize better.

This study reinforces the theoretical and empirical understanding of Batch Normalization as a cornerstone technique in modern deep learning pipelines.
