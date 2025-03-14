# Intermediate Report
**Zechen Ma, Zixi Qu, Jinyan Yi**

## Table of contents


1. [Introduction](#1-Introduction)
2. [Gradient Compression](#2-Gradient-Compression)
    2.1 [Background](#21-Background)
    2.2 [Experiment](#22-Experiment)
        2.2.1 [System Setup](#221-System-Setup) 
        2.2.1 [Compression Selection](#222-Compression-Selection) 
        2.2.1 [Experiment Setup](#223-Experiment-Setup) 
    2.3 [Conclusion](#23-Conclusion)
3. [Dynamic Bound Tolerance](#3-Dynamic-Bound-Tolerance)
    3.1 [Experiment Setup](#31-Experiment-Setup)
    3.2 [Experiment Outcomes](#32-Experiment-Outcomes)
    3.3 [Conclusion](#33-Conclusion)
4. [Tentative plan](#4-Tentative-plan)
5. [Work Distribution](#5-Work-Distribution)
6. [Conclusion](#6-Conclusion)



## 1. Introduction 
Our team based on the paper [MLT (Machine Learning Transport)](https://www.usenix.org/conference/nsdi24/presentation/wang-hao), and  researched further in 2 directions, Gradient Compression and Dynamic Bound Tolerance. The first topic was an exploration on top of the suggested compression method in the paper, data sparcification and quantization, to see if we can come up with an alternative gradient compression method that better suit the growing demand on machine learning challenges in a datacenter. The second topic, inspired by [Professor Yashar Ganjali](https://www.cs.toronto.edu/~yganjali/), was a keen idea of whether the bound tolerance factor, which was one of the three main focus in the paper, can be dynamically adjusted in a process of a training task. The motivation of both topics is to further increase the efficiency of machine learning in a datacenter, with minimal effort on updating hardware, provide a "free" update on the current datacenter setups. (For final report) In short, we have found the following ..., the experiment result can be found under [some section](#). 


## 2. Gradient Compression

### 2.1 Background
As the MLT paper suggested, gradient compression leads to lower average flow completion time (FCT). However, the mainstream compression techiniques widely adopted to the distributed machine learning community are rather trivial: quantization and sparsification. We aim to explore slightly more complex approaches with the goal of not introducing too much computational overhead on any hosts. We also want to keep the fidelity of our data as much as we can so that we don't greatly lose the model accuracy.

With the above assumptions, lossless compression is the best fit. Lemple Ziv + huffman encoding as proposed in the project proposal is a quite implementation expensive approach which doesn't satisfy the requirement of low computational overhead. It also doesn't perform well on floating point numbers (it is a popular approach widely deployed on string compression/encoding). 

On top of keeping the data fidelity and low computational overhead, we picked some ad-hoc compression technique based on what the gradident looks like. This is very application specific because different models or different dataset can result in completely different gradient distribution. The specific combination will be illustrated in the experiment section below.

### 2.2 Experiment
#### 2.2.1 System Setup

We use Python for this project development. Specifically, we have three workers, each sharing 1/3 of the dataset (MNNIST in our case). There is a server that listens for connections from the 3 workers. 3 workers are concurrently running with multi-core multi-processing in one local machine. Workers communicate with the server through TCP connections. 

Within each training iteration there will be hundreads of that in one training epoch, workers will send their respective gradient to the server. **Compression happens right before the worker sends out the gradient packets to the server**. If the server received compressed data and the current compression regime is lossless, then on the server end, there will be a corresponding **decompression happening before the server actually processes the data. Server will also need to restore the shape of the gradients if compression flattens them out and makes them lose the original shape.**

After all the workers send out either the original or compressed data and the server either purely receives or receives + decrypts/restores on the server end, the server will combine all the 3 pieces of data and perform an averaging. The averaged gradients will be sent back to all the workers to wrap up one complete activity of connection.

#### 2.2.2 Compression Selection

We started out our experiment with MNIST dataset and a fully connected simple model with no hidden layers.

During the training phases, we observed that the gradient has a lot of repetitions at the beginning and the end, and this observation applies to all the gradients across all the training phases.

With the goal of low computational overhead and preserving the data fidelity, we chose **Run Length Encoding (RLE)** to compress the gradients. How RLE works is that if it sees one gradient, e.g. 0.1261, appearing 6 times, it will compress the 6 gradients into 1 tuple as (value, count). So 6 appearances of 0.1261 becomes (0.1261, 6).

However, we realize that this approach has 2 potential bottlenecks. One is that it requires the exactly same occurences of one gradient value repeating consecutively. For example, if we see 0.1261234 and 0.1261235, RLE cannot be leveraged in this scenario. If this were indeed happening, RLE would not only deteriorate the network latency since what it does would be doubling the total network workloads (each value becomes a tuple with an extraneous argument count equal to one).

As a result, we made 2 optimizations based on the observations above. If we see a bunch of gradients that are neighboring and only has less than 1e-5 difference between each other. Then the **first optimization on top of RLE is rounding all the flattened gradients to 4 floating-point precisions**. In fact, the vast majority of gradients a lot of gradients only have 4 decimal places. So we are not losing a lot of precisions in this case.

Another optimization is that if the count of one gradient is 1, we don't represent them as a tuple. **We simply have them in its original format which will just be tensors of floating point number**

We also implemented 2 quantization techniques as baselines. One is simply cutting the precision from 32-bit to 16 bit. After researching how pytorch quantization works, this quantization doesn't just cut the precision on all the floating point numbers we see. It uses IEEE half precison floating point representation instead of single precision. So for example from the perspective of human eyes, 0.3, after this baseline quantization, will still be 0.3, but its hardware representation will be in 16 bits using the half-precision format. Both 16-bit and 32-bit are approximations to 0.3 as it cannot be perfectly represented in bianry formats. The other is normalizing all the gradients to the range of 0 and 1, and then maps them all the range of 0 and 255. That way, an 8-bit unsigned integer will be able to represent all the gradients which are originally 32-bit. These 2 approaches are both lossy.

#### 2.2.3 Experiment Setup
To showcase the performance of several compression methods mentioned above, we conduct performance tests on the same machine and collect essential data. Here's the performance data we collected:

$Model: ResNet8; Dataset: MNIST$

| Compression Method       | Total Network Latency (s) | Single Packet Size (byte) | Total Training Time(m:s) | Test Accuracy |
| ------------------------ | ------------------------- | ------------------------- | ------------------------ | ------------- |
| No compression           | 4.71                      | 831410                    | 4:03.73 total            | 0.2509        |
| RLE compression          | 43.443                    | ~ 2115100                 | 9:39.55 total            | 0.1919        |
| uint8 lossy quantization | 41.76                     | 547564                    | 7:34.48 total            | 0.1597        |
| PyTorch Quant            | 3.08                      | 417927                    | 3:58.73 total            | 0.2978        |

$Model: Single-Layer Fully Connected Neural Networ; Dataset: MNIST$

| Compression Method       | Total Network Latency (s) | Single Packet Size (byte) | Total Training Time(m:s) | Test Accuracy |
| ------------------------ | ------------------------- | ------------------------- | ------------------------ | ------------- |
| No compression           | 0.4492                    | 32120                     | 27.084  total            | 0.8960        |
| RLE compression          | 1.2519 (best case)        | 22355                     | 42.127  total            | 0.8935        |
| uint8 lossy quantization | 1.0783                    | 15892                     | 31.657  total            | 0.8920        |
| PyTorch Quant            | 0.2804                    | 16418                     | 31.651  total            | 0.8913        |

### 2.3 Conclusion 
It turns out that gradient compression is extremely ad-hoc: programmers have to manually observe the pattern of gradient distribution. For example, when figuring out the best commpression method for the single-layer fully connection NN model, we discovered that there are a lot of repetitive gradients at the beginning and the end of the gradients. This is understandable because all the edges of MNIST data are black and only the digit, the important information is white, which is in the middle. Because of this discover, we decided to employ RLE which both fully preserves data fidelity and not introduce much computational overhead.

However, the single packet size indeed got decreased from not compressing the data at all, but the total network latency still goes up. We suspect that network latency and packet size is not in a linear positive correlation - there might be other factors that we did not factor into measuring and consideration. However, reducing the packet size still achieves one of our goals that we want to lower the network bandwidth usage as much as we can. 

When it moves on to other models, where gradient distribution is completely random and we cannot observe any common patterns to help us choose what compression method would suits it the best such us ResNet8, RLE shows no improvement at all. It means that there is no one-size-fits-all lossless compression method that would universally outform pytorch qunatization for example.

We also observed that pickle dumps tensor objects much more efficiently in terms of the dumped size. If the final data type is a tensor, pickle will dump into much smaller packets compared to a data type of a list of tuples and tensors.

Regardless of model types, pytorch quantization is, unfortunately and undoubtedly, the universal approach that performs the best in terms of all the parameters we are measuring. We chose to quantize all the tensors from float32 to float16 which, again, will be changed in its binary representation from IEEE single precision to half precision.

We hate to admit that quantization (and sparsification) proposed as the mainstream approaches in the MLT paper as well as the whole machien learning community is indeed the most efficient and effective approach across models (and datasets which is to be tested, but most likely yes).


## 3. Dynamic Bound Tolerance
In the paper, it observered that distributed machine learning tasks can tolerate up to a factor $k$ of packet loss, where $k$ usually set between $0.8\% - 3\%$, without affecting the quantity and quality result at all. For example, ResNet34 set a bound-tolerance of $1\%$, that reaches the quality target with the same number of epoch as there is no packet loss. However, we observered that the bound-tolerance factor is preset before the training, and had never changed during the learning. We intuitively asked the question, can we dynamically change the bound-tolerance, so that we can reach a higher overall bound-tolerance, that potentially increases the efficient on top of the originally static bound-tolerance. To that end, we purposed a question that we would like to answer through experiment: "If we devide the training into three phases, beginning phase, midterm phase, and final phase, which phase(s) can tolerate higher packet loss, which tolerate less?"

### 3.1 Experiment Setup
<!-- This section is subject to change in final report -->
We first define the three phases. At the beginning of the training, the weight are usually set to random values, and we observer a burst of test accuracy at the beginning of the training. As the training conducts, the speed of growing test accuracy will gradually decreases in the midterm phase. Finally, during the final phase, the test accuracy will bounce between a stable number, which we refer as quality target. The test accuracy plays an important role at defining the training phases, therefore we will use it in the following way (to better explain, filled with suggested value):
```pseudo
threshold_begin = 0.08, threshold_midterm = 0.02
for each epoch:
    delta = current_test_accuracy - previous_test_accuracy
    if (delta > threshold_begin):
        phase = beginning_phase
    else if (threshold_begin > delta > threshold_midterm):
        phase = midterm_phase
    else:  # delta < threshold_midterm:
        phase = final phase
```

We would also like to define three level of bound tolerance, high, mid, and low. For experiment purpose, we temporally set high bound tolerance to $8\%$, mid bound tolerance to $2\%$, and low bound tolerance to $0.5\%$. 

To ensure the generality, we will pick several model-dataset combinations, to see if the outcome is consistent across varient of machine learning tasks. For each machine learning task, we will conduct a total number of 27 full-trainings. The number of 27 is calculated by: for each of the three phases, there are three bound tolerance level we can set, therefore, to answer our research question, a total number of $3^3 = 27$ test runs we need to conduct.

The model-dataset combinations we had picked are [This is a placehold for final report].

### 3.2 Experiment Outcomes
There has been no outcome no far but there will be ...

### 3.3 Conclusion 
There has been no conclusion no far but there will be ...



## 4. Tentative plan


| Week # | Date of the Friday | Plan |
| -------- | -------- | -------- |
| 10 | Mar 14 | Intermediate Report, Topic 2 infrastructure   |
| 11 | Mar 21 |  Conduct experiments on topic 2  |
| 12 | Mar 28 |  Continue update the experiment based on the findings in previous weeks |
| 13 | Apr 4  |  Final project presentations  |


## 5. Work Distribution
Zechen Ma (Zee):

Zixi Qu (Jessy): Help implmenting gradient compression experiment setup. Provide consultent on machine learning.

Jinyan Yi (Alex):

## 6. Conclusion

### Conclusion 1: Gradient Compression 
[TODO]

### Conclusion 2: Dynamic Bound Tolerance
[TODO]

<!-- ### 6. -->
[EOF]
