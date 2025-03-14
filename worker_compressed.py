import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import socket
import pickle
import struct
import struct  # For packing/unpacking data size
from typing import Any, Dict, List, Tuple, Set
from config import *
import time
from models import myResNet, SimpleModel
from compression import *
import asyncio

print(f"Compression Method: {compression_method}")


DEBUG_COMPRESSION = 0


class Worker:
    def __init__(self, worker_id, host="localhost", port=60000):
        self.worker_id = worker_id
        self.server_host = host
        self.server_port = port
        self.network_latency_list = []
        self.load_data()

    def calc_network_latency(self, is_send):
        self.network_latency_list.append(self.end_time - self.start_time)
        if is_send:
            print(f"Send Network latency: {self.end_time - self.start_time}")
        else:
            print(f"Recv Network latency: {self.end_time - self.start_time}")
        # reset after calculation
        self.start_time = 0
        self.end_time = 0

    def print_total_network_latency(self):
        print(
            f"Total network latency for worker {self.worker_id}: {sum(self.network_latency_list)}"
        )

    def load_data(self):
        # Load the dataloader for this worker
        with open(f"dataloader_{worker_id}.pkl", "rb") as f:
            self.dataloader = pickle.load(f)

        with open("dataloader_test.pkl", "rb") as f:
            self.test_dataloader = pickle.load(f)

    def send_data(self, sock, data):
        """Helper function to send data with a fixed-length header."""

        # Compress the data
        compressed_data = compress(data)

        if DEBUG_COMPRESSION:
            for param in compressed_data:
                print(f"Compression technique: {compression_method}")
                print(f"Compressed Gradient: {param}:")
                for weight in compressed_data[param]:
                    print()
                    print(weight)

        # Serialize the data
        data_bytes = pickle.dumps(compressed_data)
        print(f"Send data size: {len(data_bytes)}")

        # ALL NETWORK LATENCY ARE CALCULATED ON THE WORKER SIDE
        # ---------------------------------------------------------------------
        # clock starts (no compress overhead)
        self.start_time = time.perf_counter()

        # Send the size of the data first
        sock.sendall(struct.pack("!I", len(data_bytes)))

        # Send the actual data
        sock.sendall(data_bytes)

        # waiting for server response (ACK)
        ack = sock.recv(1)  # Block until acknowledgment is received
        if ack != b"A":
            raise RuntimeError("Acknowledgment not received")

        # clock ends
        self.end_time = time.perf_counter()
        self.calc_network_latency(True)
        # ---------------------------------------------------------------------

        # # print the compressed gradients
        # print(f"Worker {worker_id} sent COMPRESSED gradients {compressed_data}.")

    def recv_data(self, sock):
        """Helper function to receive data with a fixed-length header."""
        # Receive the size of the incoming data
        size_data = sock.recv(4)
        if not size_data:
            return None
        size = struct.unpack("!I", size_data)[0]

        # ---------------------------------------------------------------------
        # clock starts
        self.start_time = time.perf_counter()

        # Receive the actual data
        data = b""
        while len(data) < size:
            packet = sock.recv(size - len(data))
            if not packet:
                return None
            data += packet

        # clock ends (no decompress overhead)
        self.end_time = time.perf_counter()
        self.calc_network_latency(False)
        # ---------------------------------------------------------------------

        # Deserialize the compressed data
        compressed_data = pickle.loads(data)
        # Decompress the data using loseless quantization
        final_data = decompress(compressed_data)

        return final_data

    def send_recv(self, gradients) -> Tuple[bool, Any]:
        # Send gradients to the server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.server_host, self.server_port))
            print(f"Worker {worker_id} connected to server.")

            # Send gradients
            self.send_data(s, gradients)

            # Receive averaged gradients
            avg_gradients = self.recv_data(s)
            if avg_gradients is None:
                return (False, None)

        return (True, avg_gradients)

    def _accuracy(self, model, dataloader):
        correct, total = 0, 0
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            predicted = torch.argmax(outputs, dim=1)
            # print(f"debug1:", outputs, outputs.shape, batch_y, batch_y.shape, sep='\n')
            # correct = (outputs == batch_y).sum().item()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        self.test_accuracy = round(correct / total, 4)
        print(f"Worker {worker_id} test accuracy: {self.test_accuracy}")

    def train_worker(self):
        # Create a model
        model = SimpleModel()
        # model = myResNet()
        print(f"Model Type: {model}")

        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(5):
            for batch_X, batch_y in self.dataloader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Get gradients
                gradients = {
                    name: param.grad for name, param in model.named_parameters()
                }

                # print gradients
                if False:
                    for param in gradients:
                        print(f"the size of {param}: {len(gradients[param])}")
                        print(
                            f"the size of values wrt {param}: {type(gradients[param])}"
                        )
                        print(f"{param} before compression:")
                        for weight in gradients[param]:
                            print(type(weight))
                            print(weight)

                # Send gradients to the server
                update, avg_gradients = self.send_recv(gradients)

                if not update:
                    print(f"Worker {worker_id} failed to receive averaged gradients.")
                    continue

                if DEBUG:
                    print(
                        f"Worker {worker_id} received averaged gradients {avg_gradients}."
                    )

                # Update model parameters with averaged gradients
                for name, param in model.named_parameters():
                    param.grad = avg_gradients[name].to(param.dtype)
                optimizer.step()

            print(f"Worker {worker_id} completed epoch {epoch}")
            # test accuracy
            self._accuracy(model, self.test_dataloader)

        print(
            f"Worker {worker_id} finished training. Final test accuracy: {self.test_accuracy}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Invalid usage.")
        print("USAGE: python worker.py <WORKER_ID>")
        sys.exit(1)

    worker_id = int(sys.argv[1])  # Worker ID (0, 1, 2)
    # sleep(3)  # Wait for all the other workers to be ready

    # Create a worker
    worker = Worker(worker_id)
    worker.train_worker()
    worker.print_total_network_latency()
