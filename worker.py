import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import socket
import pickle
import struct
from time import sleep
import struct  # For packing/unpacking data size
from typing import Any, Dict, List, Tuple, Set
import time
from models import myResNet, SimpleModel

DEBUG = 0

# # some parameters for quantize_per_tensor
# quantize = True
# # 8 bits after the decimal point
# scale = 1e-8
# type = torch.float16

class Worker:
    def __init__(self, worker_id, host="localhost", port=60000):
        self.worker_id = worker_id
        self.server_host = host
        self.server_port = port
        self.network_latency_list = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_data()

    def calc_network_latency(self, is_send):
        self.network_latency_list.append(self.end_time - self.start_time)
        if is_send:
            print(f'Send Network latency: {self.end_time - self.start_time}')
        else:
            print(f'Recv Network latency: {self.end_time - self.start_time}')
        # reset after calculation
        self.start_time = 0
        self.end_time = 0

    def print_total_network_latency(self):
        print(f'Total network latency for worker {self.worker_id}: {sum(self.network_latency_list)}')

    def load_data(self):
        # Load the dataloader for this worker
        with open(f"dataloader_{self.worker_id}.pkl", "rb") as f:
            self.dataloader = pickle.load(f)

    def send_data(self, sock, data):
        """Helper function to send data with a fixed-length header."""
        # Serialize the data
        data_bytes = pickle.dumps(data)
        print(f"Send data size: {len(data_bytes)}")

        # clock starts
        self.start_time = time.perf_counter()

        # Send the size of the data first
        sock.sendall(struct.pack("!I", len(data_bytes)))

        # Send the actual data
        sock.sendall(data_bytes)

        # waiting for server response (ACK)
        ack = sock.recv(1)  # Block until acknowledgment is received
        if ack != b'A':
            raise RuntimeError("Acknowledgment not received")
        
        # clock ends
        self.end_time = time.perf_counter()
        self.calc_network_latency(True)

    def recv_data(self, sock):
        """Helper function to receive data with a fixed-length header."""
        # Receive the size of the incoming data
        size_data = sock.recv(4)
        if not size_data:
            return None
        size = struct.unpack("!I", size_data)[0]

        # clock starts
        self.start_time = time.perf_counter()

        # Receive the actual data
        data = b""
        while len(data) < size:
            packet = sock.recv(size - len(data))
            if not packet:
                return None
            data += packet

        # clock ends
        self.end_time = time.perf_counter()
        self.calc_network_latency(False)

        return pickle.loads(data)

    def send_recv(self, gradients) -> Tuple[bool, Any]:
        # Send gradients to the server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.server_host, self.server_port))
            print(f"Worker {self.worker_id} connected to server.")

            # Send gradients
            self.send_data(s, gradients)

            # print the gradients
            if DEBUG: print(f"Worker {worker_id} sent gradients {gradients}.")
            print(f"Worker {self.worker_id} sent gradients {gradients}.")

            # Receive averaged gradients
            avg_gradients = self.recv_data(s)
            print(f"Recv data size: {len(avg_gradients)}")
            if avg_gradients is None:
                return (False, None)

        return (True, avg_gradients)

    def train_worker(self):
        # Create a model
        # model = SimpleModel()
        model = myResNet().to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(5):
            for batch_X, batch_y in self.dataloader:
                # Move data to the device
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Get gradients
                gradients = {name: param.grad.cpu() for name, param in model.named_parameters()}


                # Send gradients to the server
                update, avg_gradients = self.send_recv(gradients)

                if not update:
                    print(f"Worker {self.worker_id} failed to receive averaged gradients.")
                    continue

                # Update model parameters with averaged gradients
                for name, param in model.named_parameters():
                    param.grad = avg_gradients[name].to(self.device)
                optimizer.step()

            print(f"Worker {self.worker_id} completed epoch {epoch}")

        print(f"Worker {self.worker_id} finished training.")


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
