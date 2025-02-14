# distributed-ml-training

## Usage

To set up k workers distributively train 1/k of MNIST dataset, follow these steps:
- `python3 prepare_data.py` to split MNIST into k parts where k is the number of workers
- `python3 server.py` to let server listen for connections
- set up k terminals, each runs `python3 worker.py {i}` where i is {0, 1, 2} if k = 3
- the server will do the gradient averaging and send that back every iteration

## Discovery

**2/13 IMPORTANT: FOUND GRADIENT REDUNDANCY**

Since MNIST image is flattened to 1*784, the beginning and the end of the image are both -1 (black color?), the gradients on those areas are also the SAME

That way, the next step will be to find **loseless compression algorithm that are good at leveraging REDUNDANCY**

AND clock the 2 approaches, compare the latency
