## Neural Network Acceleration Study Season #2
This is a repository of the study "neural network acceleration". The goal of this study is to understand the acceleration of nerual networks on various devices. The topic of acceleration includes `CPU`,`GPU`, `FPGA`, `ASIC` , `NPU` and `PIM`. Our materials are open to this github and youtube.

#### CPU/GPU and NPU
- Desinging optimized BLAS for CPU or GPU
- Optimal primitive selection on heterogeneous system architecture (HSA) device
- CUDA/OpenCL kernel design

#### ASIC and FPGA
- Low-power inference acceleration using HLS or RTL design
- High computing performance training accelerator

#### PIM (NDP)
- DIMM and HMC based neural acceleration system
- Non-HBM based design

## Paper List (17)
### Processor based Acceleration (9)
	CPU, GPU, and special system based acceleration (Parallel computing, Distribution computing)
	1. AccUDNN: A GPU Memory Efficient Accelerator for Training Ultra-deep Neural Networks, arxiv, 2019.
	2. Zion: Facebook Next-Generation Large-memory Unified Training Platform, HotChips, 2019.
	3. µLayer:Low Latency On-Device Inference Using Cooperative Single-Layer Acceleration and Processor-Friendly Quantization, EuroSys, 2019.
	4. Scalpel: Customizing DNN pruning to the underlying hardware parallelism, ISCA, 2017.
	5. MOSAIC: Heterogeneity-, Communication-, and Constraint-Aware Model Slicing and Execution for Accurate and Efficient Inference, PACT, 2019.
	6. Optimal DNN Primitive Selection with Partitioned Boolean quadratic Programming, ACM CGO, 2019.
	7. Neural Network Inference on Mobile SoCs, Arxiv 2019.
	8. Learning to infer: RL-based search for DNN primitive selection on Heterogeneous Embedded Systems, DATE, 2019.
	9. Performance analysis of CNN frameworks for GPUs, ISPASS, 2018.
  

### ASIC and FPGA (6)
	1. Cambricon: An instruction set architecture for neural networks, ISCA, 2016.
	2. In-Datacenter Performance Analysis of a Tensor Processing Unit, ISCA, 2017.
	3. Overcoming Data Transfer Bottlenecks in FPGA-based DNN Accelerators via Layer Conscious Memory Management, DAC, 2019.
	4. Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks, FPGA, 2015.
	5. FA3C: FPGA-Accelerated Deep Reinforcement Learning, ASPLOS, 2019.
	6. Cambricon-S: Addressing Irregularity in Sparse Neural Networks through A Cooperative Software/Hardware Approach, MICRO, 2018.
### PIM & NDP (2)
	1. Processing-in-Memory for Energy-efficient Neural Network Training: A Heterogeneous Approach, MICRO, 2018.
	2. TensorDIMM: A Practical Near-Memory Processing Architecture for Embeddings and Tensor Operations in Deep Learning, MICRO, 2019.

   
## Presentation with Video
### Week1: Introduction of Neural network acceleration (February 02, 2020)
**Optimal DNN Primitive Selection with Partitioned Boolean quadratic Programming**  

	Presenter: Constant Park (http://esoc.hanyang.ac.kr/people/sangsoo_park/index.html)  
	PPT: https://github.com/ConstantPark/Nerual-Network-Acceleration/blob/master/Optimal%20DNN%20Primitive%20Selection%20with%20Partitioned%20Boolean%20Quadratic%20Programming.pdf   
	Video: https://youtu.be/ZLGLogU5mt0   

  

### Week2: HW accelerator (ASIC) and GPU acceleration  (February 16, 2020)
**Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks**  

	Presenter: 김태완 (rlaxodhksk@snu.ac.kr)  
	PPT: https://github.com/ConstantPark/Nerual-Network-Acceleration/blob/master/Optimizing%20FPGA-based%20Accelerator%20Design%20for%20Deep%20Convolutional%20Neural%20Networks%20Chen%20Zhang%20et%20al%20-%20Louis%20tw%20Kim%20Presentation.pdf   
	Video: https://youtu.be/tgB_o4E9PSw  


### Week3: CPU/GPU acceleration (March 8, 2020)
**Performance analysis of CNN frameworks for GPUs**  

	Presenter: Martin (dhhwang89@gmail.com)
	PPT: https://github.com/ConstantPark/Nerual-Network-Acceleration/blob/master/Performance_Analysis_of_CNN_Frameworks_for_GPUs.pdf  
	Video: https://youtu.be/6LIalb6nEqE    
	
**µLayer:Low Latency On-Device Inference Using Cooperative Single-Layer Acceleration and Processor-Friendly Quantization**   
	
	Presenter: Martin (dhhwang89@gmail.com)  
	PPT: https://github.com/ConstantPark/Nerual-Network-Acceleration/blob/master/uLayer_%20Low_Latency_On-Device_Inference_Using_Cooperative_Single-Layer_Acceleration_and_Processor-Friendly_Quantization.pdf  
	Video: https://youtu.be/ofHqG2z-X4Q   

### Week4: CPU/GPU acceleration and Systolic Accelerator (March 15, 2020)
**Scalpel: Customizing DNN pruning to the underlying hardware parallelism**  

	Presenter: DownyK (TeamBehindDowny@gmail.com)  
	PPT: https://github.com/ConstantPark/Nerual-Network-Acceleration/blob/master/Scalpel_Customizing%20DNN%20pruning%20to%20the%20underlying%20hardware%20parallelism%2C.pdf   
	Video: https://youtu.be/z0Jy8vhZT38 

**Gemmini: An Agile Systolic Array Generator Enabling Systematic Evaluations of Deep-Learning Architecturesr**  

	Presenter: Constant Park (sonicstage12@naver.com)  
	PPT: https://github.com/ConstantPark/Nerual-Network-Acceleration/blob/master/Gemmini-%20An%20Agile%20Systolic%20Array%20Generator%20Enabling%20Systematic%20Evaluations%20of%20Deep-Learning%20Architectures.pdf
	Video: https://youtu.be/nqDLiLjySLE

### Week5: HW accelerator (ASIC) and CPU/GPU acceleration (March 29, 2020)
**MOSAIC: Heterogeneity-, Communication-, and Constraint-Aware Model Slicing ~ Efficient Inference**  
	
	Presenter: 이제민 (leejaymin@cnu.ac.kr)  
	PPT:   
	Video:   

**In-Datacenter Performance Analysis of a Tensor Processing Unit**    
	
	Presenter: Constant Park (sonicstage12@naver.com)
	PPT:   
	Video:   

### Week6: CPU/GPU acceleration and HW accelerator (FPGA) (April 05, 2020)
**Neural Network Inference on Mobile SoCs**   
	
	Presenter: 전지혜 (jyeah05@gmail.com)  
	PPT:   
	Video:   

**FA3C: FPGA-Accelerated Deep Reinforcement Learning**    
	
	Presentoe: 김석중 (rlatjrwnd242@naver.com)  
	PPT:   
	Video:   

### Week7: NDP (Near-data Processing) and CPU/GPU acceleration (April 19, 2020)
**TensorDIMM: A Practical Near-Memory Processing Architecture for Embeddings and Tensor Operations in Deep Learning**    
	
	Presenter: Constant Park (sonicstage12@naver.com)
	PPT:   
	Video:   

**Zion: Facebook Next-Generation Large-memory Unified Training Platform**   
	
	Presenter: DownyK (TeamBehindDowny@gmail.com)
	PPT:   
	Video:   


### Week8: HW accelerator (FPGA) and CPU/GPU acceleration (May 03, 2020)
**Overcoming Data Transfer Bottlenecks in FPGA-based DNN Accelerators via Layer Conscious Memory Management**   
	
	Presenter: 이제민 (leejaymin@cnu.ac.kr)
	PPT:   
	Video:  

**Learning to infer: RL-based search for DNN primitive selection on Heterogeneous Embedded Systems**   
	
	Presenter: 전지혜 (jyeah05@gmail.com)
	PPT:   
	Video:   

**Cambricon-S: Addressing Irregularity in Sparse Neural Networks through A Cooperative Software/Hardware Approachh**    
	
	Presenter: 김석중 (rlatjrwnd242@naver.com)  
	PPT:   
	Video:   
	
## Contributors
**Main Contributor**: Constant Park (sonicstage12@naver.com)  
**Presenters**: Constanr Park (sonicstage12@naver.com), 이제민 (leejaymin@cnu.ac.kr), 김태완 (rlaxodhksk@snu.ac.kr), DownyK (TeamBehindDowny@gmail.com), 전지혜 (jyeah05@gmail.com), Martin (dhhwang89@gmail.com), 김용우 (guruzoa@gmail.com), 
(rlatjrwnd242@naver.com)

