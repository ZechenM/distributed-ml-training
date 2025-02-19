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


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)  # MNIST images are 28x28

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        return self.fc(x)
    

class myResNet(nn.Module):
    def __init__(self):
        super(myResNet, self).__init__()
        ## COMPLETED ##
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.active1 = nn.ReLU()

        # Residual Unit
        self.layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.active2 = nn.ReLU()
        self.layer2_5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')  # layer 2.5

        # Residual Unit end
        self.active_residual = nn.ReLU()

        ##### Change up until this point.
        self.PL3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.active4 = nn.ReLU()

        self.layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.active5 = nn.ReLU()

        self.PL6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer7 = nn.Linear(in_features=1024, out_features=128)
        self.active7 = nn.ReLU()

        self.layer8 = nn.Linear(in_features=128, out_features=10)
        self.active8 = nn.Sigmoid()


    def forward(self, x):
        x_dim = x.dim()
        x = self.active1(self.layer1(x))

        x1 = x  # store for skip connection

        # Residual Unit
        x = self.layer2(x)
        x = self.active2(x)
        x = self.layer2_5(x)
        x2 = x  # store for skip connection
        # Residual Unit end
        x = x1 + x2
        x = self.active_residual(x)
        ##### Change up until this point.


        x = self.PL3(x)

        x = self.active4(self.layer4(x))
        x = self.active5(self.layer5(x))
        x = self.PL6(x)

        # x = torch.flatten(x)
        # Ask ChatGPT: if input dim is 3, flat to 1D, if input dim is 4, eg, [100, xxx,xxx,xxx], flatten to [100, something]
        # ChatGPT generated code
        if x_dim == 4:  # For 4D input, flatten to [batch_size, -1]
            x = torch.flatten(x, start_dim=1)
        elif x_dim == 3:  # For 3D input, flatten to 1D
            x = torch.flatten(x)
        # ChatGPT Done
        
        x = self.active7(self.layer7(x))
        x = self.active8(self.layer8(x))
        return x  ## COMPLETED ##


class Worker:
    def __init__(self, worker_id, host="localhost", port=60000):
        self.worker_id = worker_id
        self.server_host = host
        self.server_port = port
        self.network_latency_list = []
        self.load_data()

    def calc_network_latency(self):
        self.network_latency_list.append(self.end_time - self.start_time)
        print(f'Network latency: {self.end_time - self.start_time}')
        # reset after calculation
        self.start_time = 0
        self.end_time = 0

    def print_total_network_latency(self):
        print(f'Total network latency for worker {self.worker_id}: {sum(self.network_latency_list)}')

    def load_data(self):
        # Load the dataloader for this worker
        with open(f"dataloader_{worker_id}.pkl", "rb") as f:
            self.dataloader = pickle.load(f)

    def send_data(self, sock, data):
        """Helper function to send data with a fixed-length header."""
        # Serialize the data
        data_bytes = pickle.dumps(data)

        # clock starts
        self.start_time = time.perf_counter()
        
        # Send the size of the data first
        sock.sendall(struct.pack("!I", len(data_bytes)))

        # Send the actual data
        sock.sendall(data_bytes)


    def recv_data(self, sock):
        """Helper function to receive data with a fixed-length header."""
        # Receive the size of the incoming data
        size_data = sock.recv(4)
        if not size_data:
            return None
        size = struct.unpack("!I", size_data)[0]

        # Receive the actual data
        data = b""
        while len(data) < size:
            packet = sock.recv(size - len(data))
            if not packet:
                return None
            data += packet
        
        # clock ends
        self.end_time = time.perf_counter()
        self.calc_network_latency()
        
        return pickle.loads(data)


    def send_recv(self, gradients) -> Tuple[bool, Any]:
        # Send gradients to the server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.server_host, self.server_port))
            print(f"Worker {worker_id} connected to server.")

            # Send gradients
            self.send_data(s, gradients)

            # print the gradients
            print(f"Worker {worker_id} sent gradients {gradients}.")

            # Receive averaged gradients
            avg_gradients = self.recv_data(s)
            if avg_gradients is None:
                return (False, None)

        return (True, avg_gradients)


    def train_worker(self):
        # Create a model
        # model = SimpleModel()
        model = myResNet()
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
                gradients = {name: param.grad for name, param in model.named_parameters()}

                # Print the size of gradients
                for name, grad in gradients.items():
                    print(f"Gradient size for {name}: {grad.size()}")

                # Send gradients to the server
                update, avg_gradients = self.send_recv(gradients)

                if not update:
                    print(f"Worker {worker_id} failed to receive averaged gradients.")
                    continue

                print(f"Worker {worker_id} received averaged gradients {avg_gradients}.")

                # Update model parameters with averaged gradients
                for name, param in model.named_parameters():
                    param.grad = avg_gradients[name]
                optimizer.step()

            print(f"Worker {worker_id} completed epoch {epoch}")

        print(f"Worker {worker_id} finished training.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Invalid usage.")
        print("USAGE: python worker.py <WORKER_ID>")
        sys.exit(1)
    
    worker_id = int(sys.argv[1])  # Worker ID (0, 1, 2)
    sleep(3)  # Wait for all the other workers to be ready

    # Create a worker
    worker = Worker(worker_id)
    worker.train_worker()
    worker.print_total_network_latency()
