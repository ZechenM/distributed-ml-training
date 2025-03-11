# distributed-ml-training

## Usage

The training is orchestrated by shell script:

1. Config the experiment, go to `config.py`.

2.  `time ./startAll.sh`. This will display total network latency and test accuracy

3. To view full test accuracy history, `cat ./logs/{last_experiment}/worker* | grep "test accuracy"`


If first time, data is not prepared, it will run the data generation script. After the data is generated, re-run the script.

The shell script do error handling and self-cleanup on exit.

Server and Workers print message can be found under `logs/`


## Discovery

### (3) [date] [title]
[description]

### (2) 2/16 added RLE encoding compression technique

Pure Network latency (excludes compression overhead) decreases when gradients are compressed first and then sent out

Would be worth comparing RLE compression with quantization and combining those two techniques to further minimize network latency

### (1) 2/13 IMPORTANT: FOUND GRADIENT REDUNDANCY

Since MNIST image is flattened to 1*784, the beginning and the end of the image are both -1 (black color?), the gradients on those areas are also the SAME

That way, the next step will be to find **loseless compression algorithm that are good at leveraging REDUNDANCY**

AND clock the 2 approaches, compare the latency


