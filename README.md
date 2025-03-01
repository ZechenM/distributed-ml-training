# distributed-ml-training

## Usage

Update: the training is now orchestrated by shell script. Old usage is below new usage.

run `./startAll.sh` for normal server and worker

`./startAll.sh -c` for server and worker with RLE gradient compression

If first time, data is not prepared, it will run the data generation script. After the data is generated, re-run the script.

The shell script do error handling and self-cleanup on exit.

Server and Workers print message can be found under `logs/`

---

**Old usage**

To set up k workers distributively train 1/k of MNIST dataset, follow these steps:

- `python3 prepare_data.py` to split MNIST into k parts where k is the number of workers
- `python3 server.py` to let server listen for connections
- set up k terminals, each runs `python3 worker.py {i}` where i is {0, 1, 2} if k = 3
- the server will do the gradient averaging and send that back every iteration

## Discovery

### 2/13 IMPORTANT: FOUND GRADIENT REDUNDANCY

Since MNIST image is flattened to 1*784, the beginning and the end of the image are both -1 (black color?), the gradients on those areas are also the SAME

That way, the next step will be to find **loseless compression algorithm that are good at leveraging REDUNDANCY**

AND clock the 2 approaches, compare the latency

## 2/16 added RLE encoding compression technique

Pure Network latency (excludes compression overhead) decreases when gradients are compressed first and then sent out

Would be worth comparing RLE compression with quantization and combining those two techniques to further minimize network latency
