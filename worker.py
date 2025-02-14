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


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)  # MNIST images are 28x28

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        return self.fc(x)


def send_data(sock, data):
    """Helper function to send data with a fixed-length header."""
    # Serialize the data
    data_bytes = pickle.dumps(data)
    # Send the size of the data first
    sock.sendall(struct.pack("!I", len(data_bytes)))
    # Send the actual data
    sock.sendall(data_bytes)


def recv_data(sock):
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
    return pickle.loads(data)

def send_recv(host, port, gradients) -> Tuple[bool, Any]:
    # Send gradients to the server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print(f"Worker {worker_id} connected to server.")

        # Send gradients
        send_data(s, gradients)
        
        # print the gradients
        print(f"Worker {worker_id} sent gradients {gradients}.")
        
        # Receive averaged gradients
        avg_gradients = recv_data(s)
        if avg_gradients is None:
            return (False, None)
        
    return (True, avg_gradients)


def train_worker(worker_id, dataloader, port=60000, host="localhost"):
    # Create a model
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(5):
        for batch_X, batch_y in dataloader:
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
            update, avg_gradients = send_recv(host, port, gradients) 

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

    # Load the dataloader for this worker
    with open(f"dataloader_{worker_id}.pkl", "rb") as f:
        dataloader = pickle.load(f)

    sleep(3)  # Wait for all the other workers to be ready

    train_worker(worker_id, dataloader)
