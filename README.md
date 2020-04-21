## Neural Network Acceleration Study Season #2
This is a repository of the study "neural network acceleration". The goal of this study is to understand the acceleration of nerual networks on various devices. The topic of acceleration includes `CPU`, `GPU`, `NPU`, `ASIC`, `FPGA`, and `NDP`. Our materials are open to this github and youtube. This study is supported by Facebook community, "AI Robitcs Korea".

#### CPU/GPU, NPU, and distributed computing
- Fast acceleration of inference/training on general processor (CPU/GPU)
- Distributed computing for large training system
- Heterogeneous system architecture (HSA) device

#### ASIC and FPGA
- Low-power inference acceleration using RTL/HLS design
- High computing performance interfence/training accelerator

#### Near-data Processing (NDP)
- Data processing unit for neural network acceleration (w/o HBM based accelerator)

## Paper List (21)
### CPU/GPU/NPU based Acceleration (14)
	1. Capuchin: Tensor-based GPU Memory Management for Deep Learning, ASLPOS, 2020
	2. Parallax: Sparsity-aware Data Parallel Training of Deep Neural Networks, EuroSys, 2019
	3. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism, NIPS, 2019
	4. DeepRebirth: Accelerating Deep Neural Network Execution on Mobile Devices, AAAI, 2018
	5. OC-DNN: Exploiting Advanced Unified Memory Capabilities in CUDA 9 and Volta GPUs for Out-of-Core DNN Training, HiPC, 2018
	6. Acorns: A Framework for Accelerating Deep Neural Networks with Input Sparsity, PACT, 2019
	7. Edge AI: On-Demand Accelerating Deep Neural Network Inference via Edge Computing, IEEE TWC, 2019
	8. Balanced Sparsity for Efficient DNN Inference on GPU, AAAI, 2019
	9. DWM: A Decomposable Winograd Method for Convolution Acceleration, AAAI, 2020
	10. Split-CNN: Splitting Window-based Operations in Convolutional Neural Networks for Memory System Optimization, ASLOPS, 2020
	11. PatDNN: Achieving Real-Time DNN Execution on Mobile Devices with Pattern-based Weight Pruning, ASLOPS, 2020
	12. FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System, ASLOPS, 2020
	13. Neural Network Inference on Mobile SoC, arXiv, 2020
	14. Learning to infer: RL-based search for DNN primitive selection on Heterogeneous Embedded Systems, DATE, 2019
### Dedicated neural network accelerator (5)
	1. Caffeine: Toward Uniformed Representation and Acceleration for Deep Convolutional Neural Networks, IEEE TCAD, 2019
	2. MnnFast: a fast and scalable system architecture for memory-augmented neural networks, ISCA, 2019
	3. GANAX: A unified mimd-simd acceleration for generative adversarial networks, ISCA, 2018
	4. Deep Learning Acceleration with Neuron-to-Memory Transformation, HPCA, 2018
	5. FA3C: FPGA-Accelerated Deep Reinforcement Learning, ASPLOS, 2019.

### NDP (1)
	1. TensorDIMM: A Practical Near-Memory Processing Architecture for Embeddings and Tensor Operations in Deep Learning, MICRO, 2019

### Benchmark (1)
	1. MLPerf: An Industry Standard Benchmark Suite for Machine Learning Performance, MICRO, 2020
	
## Presentation with Video
### Week1: Introduction (April 14, 2020)
**GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism**

	Presenter: Constant Park (http://esoc.hanyang.ac.kr/people/sangsoo_park/index.html)  
	PPT: https://github.com/ConstantPark/Neural-Network-Acceleration-2/blob/master/GPipe_Easy%20Scaling%20with%20Micro%20Batch%20Pipeline%20Parallelism.pdf
	Video: https://youtu.be/jIW4zoF0pOo

### Week2: Neural Acceleration on SoC and  (April 28, 2020)
**Neural Network Inference on Mobile SoC**

	Presenter: 전지예 ()  
	PPT: 
	Video: 
	
**Capuchin: Tensor-based GPU Memory Management for Deep Learning**

	Presenter: 문정우 ()  
	PPT: 
	Video: 


## Contributors
**Main Contributor**: Constant Park (sonicstage12@naver.com), Louis Lee (louislee111@naver.com), 이재윤 (v2fds@naver.com), Hyuntak Lim (loo3944@naver.com), Yongwoo Kim (yongwoo.kim@smu.ac.kr)

**Presenters**: Constant Park (sonicstage12@naver.com), Louis Lee (louislee111@naver.com), 이재윤 (v2fds@naver.com), Hyuntak Lim (loo3944@naver.com), Yongwoo Kim (yongwoo.kim@smu.ac.kr)
