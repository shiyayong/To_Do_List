# To_Do_List

#

# 2024年08月26日
Efficient Joint Optimization of Layer-Adaptive Weight Pruning in Deep Neural Networks  层剪枝的，最好是能够运行出来，这个模型剪枝90%而任务准确率降低了2%，那么我们能不能做到，简直90% 任务1的准确率下降2% 而任务2的准确率不下降呢？ 先调通再想办法



下面都是拆分计算-边缘智能的A会A刊的论文。

1 Efficient Parallel Split Learning over Resource-constrained Wireless Edge Networks

2 Edge-MSL: Split Learning on the Mobile Edge via Multi-Armed Bandits

3 Split Learning Over Wireless Networks: Parallel Design and Resource Management

4 An Efficient Light-weight Vision Transformer Based on Split-and-Merge Strategy and Channel Shuffle

5 Edge-MSL: Split Learning on the Mobile Edge via Multi-Armed Bandits

6 Splitwise: Efficient Generative LLM Inference Using Phase Splitting

# 2024年08月21日
DAC论文搜索关键词

协同感知 Collaborative Perception

空天地一体化网络Space-Integrated-Ground

SAR 图像 关联网站 https://www.sartechnology.ca/imagerecognition/

高清搜救杆式摄像机: High-definition Search and Rescue Pole Camera

头盔式热像仪、头盔摄像机: Helmet-mounted Thermal Imager, Helmet Camera

受害者探测器（携带摄像头）: Victim Detector (with Camera), USAR Life Detectors/ Search Cameras 

多传感器生物望远镜: Multi-sensor Biometric Binoculars

直升飞机中远距离成像: Helicopter Mid-Range and Long-Range Imaging

无人机: Drone or Unmanned Aerial Vehicle (UAV)

携带摄像头的搜救犬 

搜救机器人

导航机器人


# 2024年08月17日 任务优先级 + 自适应张量压缩 搜索关键词： task-specific optimization  , task-specific decoders,

Quantifying Task Priority for Multi-Task Optimization  暂时没有代码

AdaMTL: Adaptive Input-dependent Inference for Effcient Multi-Task Learning 有代码，试着调一下，在一个完整多任务骨架上，生成了一个策略网络：针对特定任务，决定网络的各个层有哪些是有利于该任务计算的，则激活，不利于该任务计算的，则使其沉默。另外，将一个图像分为不同的块，给每一个图像块做一个任务关联标记，明确计算任务和块的关系，这样让模型在执行相关任务的时候就只接收样本的部分特征，在很大程度上避免了复杂的冗余计算。

TaskExpert: Dynamically Assembling Multi-Task Representations with Memorial Mixture-of-Experts 挺好的，但是没有代码，已经给作者发邮件了。

MTLoRA: A Low-Rank Adaptation Approach for Efficient Multi-Task Learning 关联度比较高，正在着手尝试。

Generic-to-Specific Distillation of Masked Autoencoders 从通用到特定任务的大小模型之间的蒸馏
# 2024年08月14日 多任务拆分计算

Performance-aware Approximation of Global Channel Pruning for Multitask CNNs 多任务剪枝

STEM: Unleashing the Power of Embeddings for Multi-task Recommendation 多任务计算，内存不够，没调通。

Controllable Dynamic Multi-Task Architectures 可控动态多任务架构 还没调，可以调调试试，没有代码。

#   paper 剪枝、蒸馏 2024年08月10日  # 关键词：Task-specific distillation edge

Efficient Stitchable Task Adaptation

很有意思，但是不是三个研究点里边的内容。PASS: Patch Automatic Skip Scheme for Efficient

On-Device Video Perception

Energy-Efficient Optimal Mode Selection for Edge AI Inference via Integrated Sensing-Communication-Computation

是研究点1 数据压缩的关联文章   Task-aware Distributed Source Coding under Dynamic Bandwidth    
DiSparse: Disentangled Sparsification for Multitask Model Compression

eTag: Class-Incremental Learning via Hierarchical Embedding Distillation and Task-Oriented 确认有代码 还没调过 打算调  ☆   2024年08月14日更新:success

Generic-to-Specific Distillation of Masked Autoencoders ☆ 确认有代码 还没调过  打算调  ☆ 调需要时间较久

DiSparse: Disentangled Sparsification for Multitask Model Compression 有代码可以调，缺点在于优点老，通了。！ ☆ 

DOT: A Distillation-Oriented Trainer

Multi-Task Learning with Knowledge Distillation for Dense Prediction  没有代码

Less is More: Task-aware Layer-wise Distillation for Language Model Compression 正在调 2024年08月14日更新：fail 数据集是代码自动下载的，但是下载代码一直报错。

TinyTrain: Resource-Aware Task-Adaptive Sparse Training of DNNs at the Data-Scarce Edge #20248月底公布代码

AdaMV-MoE: Adaptive Multi-Task Vision Mixture-of-Experts ☆

Quantifying Task Priority for Multi-Task Optimization

TaskFusion: An Efficient Transfer Learning Architecture with Dual Delta Sparsity for Multi-Task Natural Language Processing

ТА2P: TASK-AWARE ADAPTIVE PRUNING METHOD FOR IMAGE CLASSIFICATION ON EDGE DEVICES

MTL-Split: Multi-Task Learning for Edge Devices using Split Computing

DTMM: Deploying TinyML Models on Extremely Weak IoT Devices with Pruning # 作者已给代码 https://github.com/LixiangHan/DTMM

MTL-Split: Multi-Task Learning for Edge Devicesusing Split Computing # 有代码已经调通

Neural Rate Estimator and Unsupervised Learning for Efficient Distributed Image Analytics in Split-DNN models****

Update Compression for Deep Neural Networks on the Edge # 端边协同推理

TIES-Merging: Resolving Interference When Merging Models 模型合并蒸馏

AdapMTL: Adaptive Pruning Framework for Multitask Learning Model # 尚未有代码 关联性比较高


# paper 多专家 2024年08月07日

One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation



Class Attention Transfer Based Knowledge Distillation

Knowledge amalgamation for object detection with transformers  已有调试结果，初步可行

Curriculum Temperature for Knowledge Distillation

From Knowledge Distillation to Self-Knowledge Distillation: A Unified Approach with Normalized Loss and Customized Soft Labels

Efficient Multitask Dense Predictor via Binarization

MIND: Multi-Task Incremental Network Distillation 调试失败

Class Incremental Learning with Multi-Teacher Distillation 无代码

Knowledge Distillation and Training Balance for Heterogeneous Decentralized Multi-Momodal Learning over Wireless Networks 适合参考不适合代码复现 比较麻烦

A Teacher-Free Graph Knowledge Distillation Framework With Dual Self-Distillation 没调通

Efficient Deweahter Mixture-of-Experts with Uncertainty-Aware Feature-Wise Linear Modulation 没代码

具有不确定性感知功能的高效Deweahter混合专家线性调制

Demystifying Softmax Gating Function in Gaussian Mixture of Experts 没有代码


MoDE: A Mixture-of-Experts Model with Mutual Distillation among the Experts 没调通



Multi-Task Dense Prediction via Mixture of Low-Rank Experts 数据集70G可能需要的运行时间比较久 所以没尝试



MoME: Mixture-of-Masked-Experts for Efficient Multi-Task Recommendation ☆






# Paper 2024年08月01日

Quantifying Task Priority for Multi-Task Optimization 

Cloud-Device Collaborative Learning for Multimodal Large Language Models

Asymmetric Masked Distillation for Pre-Training Small Foundation Models

Bootstrapping SparseFormers from Vision Foundation Models

PeLK: Parameter-efficient Large Kernel ConvNets with Peripheral Convolution

PeerAiD: Improving Adversarial Distillation from a Specialized Peer Tutor

Omni-SMoLA: Boosting Generalist Multimodal Models with Soft Mixture of Low-rank Experts

OmniGlue: Generalizable Feature Matching with Foundation Model Guidance

OMG-Seg: Is One Model Good Enough For All Segmentation?

Multi-Task Dense Prediction via Mixture of Low-Rank Experts

Jack of All Tasks Master of Many: Designing General-Purpose Coarse-to-Fine Vision-Language Model

？ Reconstruction-free Cascaded Adaptive Compressive Sensing 

Retraining-Free Model Quantization via One-Shot Weight-Coupling Learning

Rethinking Multi-view Representation Learning via Distilled Disentangling

Retraining-Free Model Quantization via One-Shot Weight-Coupling Learning

Training-Free Pretrained Model Merging ☆ 

UniPT: Universal Parallel Tuning for Transfer Learning with Efficient Parameter and Memory ☆ 

UniPTS: A Unified Framework for Proficient Post-Training Sparsity

Your Student is Better Than Expected: Adaptive Teacher-Student Collaboration for Text-Conditional Diffusion Models

OTOV2: Automatic, Generic, User-Friendly

Auto-Train-Once: Controller Network Guided Automatic Network Pruning from Scratch

Network Expansion for Practical Training Acceleration

Get More at Once: Alternating Sparse Training with Gradient Correction

PTQ4SAM: Post-Training Quantization for Segment Anything

A Fast Post-Training Pruning Framework for Transformers

Retraining-free Model Quantization via One-ShotWeight-Coupling Learning

TinySAM: Pushing the Envelope for Efficient Segment Anything Model

Learning Pruning-Friendly Networks Via Frank-Wolfe: One-Shot, Any-Sparsity, And No Retraining

Train Once, and Decode As You Like

Get More at Once: Alternating Sparse Training with Gradient Correction

Structurally Prune Anything: Any Architecture, Any Framework, Any Time

E2E-AT: A Unified Framework for Tackling Uncertainty in Task-Aware End-to-End Learning

DeepSpeed Data Effciency: Improving Deep Learning Model Quality and Training Effciency via Effcient Data Sampling and Routing

MIND: Multi-Task Incremental Network Distillation

Data Shunt: Collaboration of Small and Large Models for Lower Costs and Better Performance

[no code] EPSD:Early Pruning with Self-Distillation for Efficient Model Compression

[no code] One Step Learning, One Step Review

Partial Label Learning with a Partner

Towards Real-World Test-Time Adaptation: Tri-net Self-Training with Balanced Normalization

Amalgamating Multi-Task Models with Heterogeneous Architectures ☆ 尝试过，没成功

# 微调

[no code] Measuring Task Similarity and Its Implication in Fine-Tuning Graph Neural Network



# 教师-多任务-模型

Class Incremental Learning with Multi-Teacher Distillation

Mod-Squad: Designing Mixtures of Experts As Modular Multi-Task Learners ☆

AAAI24' FedCD: Federated Semi-Supervised Learning with Class Awareness Balance via Dual Teachers

AAAI24' Let All Be Whitened: Multi-Teacher Distillation for Efficient Visual Retrieval

SAM-PARSER: Fine-Tuning SAM Efficiently by Parameter Space Reconstruction

Let All Be Whitened: Multi-Teacher Distillation for Efficient Visual Retrieval

Collaborative Consortium of Foundation Models for Open-World Few-Shot Learning

Say Anything with Any Style

How to Trade Off the Quantity and Capacity of Teacher Ensemble: Learning Categorical Distribution to Stochastically Employ A 

Teacher for Distillation 

[no code] Efficient Deweahter Mixture-of-Experts with Uncertainty-Aware Feature-Wise Linear Modulation

[no code] Learning Multi-Task Sparse Representation Based on Fisher Information

[no code]  Teacher as a Lenient Expert: Teacher-Agnostic Data-Free Knowledge Distillation

# 架构搜索论文

Boosting Order-Preserving and Transferability for Neural Architecture Search: a Joint Architecture Refined Search and Fine-tuning Approach

Building Optimal Neural Architectures using Interpretable Knowledge

Towards Accurate and Robust Architectures via Neural Architecture Search



# 来自aaai2024 有点关联的论文

E2E-AT: A Unified Framework for Tackling Uncertainty in Task-Aware End-to-End Learning

剪枝论文列表：https://github.com/ghimiredhikura/Awasome-Pruning

auc问题：AUC Optimization from Multiple Unlabeled Datasets


