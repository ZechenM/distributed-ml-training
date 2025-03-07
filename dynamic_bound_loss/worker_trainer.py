import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import socket
import pickle
import struct
import time
from transformers import Trainer, TrainingArguments
from torchvision import transforms
import torchvision.models as models
import random
import numpy as np
import os
from sklearn.metrics import accuracy_score
from torch.serialization import safe_globals

class CIFAR100Dataset(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.images = []
        self.labels = []
        
        # Extract images and labels from the subset
        for img, label in subset:
            self.images.append(img)
            self.labels.append(label)
            
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        # Debug print to verify data
        img = self.images[idx]
        label = self.labels[idx]
        
        # Ensure proper tensor formats - don't rewrap tensors
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float)
        else:
            # Ensure it's a float tensor
            img = img.float()
            
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
            # Ensure it's a long tensor for labels
            label = label.long()
            
        print(f"Returning data for index {idx}: {type(img)}, shape: {img.shape}")
        
        return {
            'pixel_values': img,
            'labels': label
        }


class DistributedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Extract our custom parameters before passing to parent
        self.server_host = kwargs.pop('server_host', 'localhost')
        self.server_port = kwargs.pop('server_port', 60000)
        self.worker_id = kwargs.pop('worker_id', 0)
        self.device = kwargs.pop('device', torch.device("cpu"))  # Get device from kwargs
        self.network_latency_list = []
        self.start_time = 0
        self.end_time = 0
        
        # Initialize parent class with remaining arguments
        super().__init__(*args, **kwargs)

    def send_data(self, sock, data):
        data_bytes = pickle.dumps(data)
        self.start_time = time.perf_counter()
        sock.sendall(struct.pack("!I", len(data_bytes)))
        sock.sendall(data_bytes)
        self.end_time = time.perf_counter()
        self.calc_network_latency(True)

    def recv_data(self, sock):
        size_data = sock.recv(4)
        if not size_data:
            return None
        size = struct.unpack("!I", size_data)[0]
        
        self.start_time = time.perf_counter()
        data = b""
        while len(data) < size:
            packet = sock.recv(size - len(data))
            if not packet:
                return None
            data += packet

        self.end_time = time.perf_counter()
        self.calc_network_latency(False)
        return pickle.loads(data)

    def send_recv(self, gradients):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.server_host, self.server_port))
            print(f"Worker {self.worker_id} connected to server.")
            self.send_data(s, gradients)
            avg_gradients = self.recv_data(s)
            if avg_gradients is None:
                return False, None
        return True, avg_gradients

    def get_train_dataloader(self):
        """
        Override the default dataloader to use our custom collate function
        """
        print("Using custom get_train_dataloader")
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=custom_collate_fn,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def training_step(self, model, inputs, num_items_in_batch):
        """
        Override the training step to implement distributed training
        Args:
            model: The model to train
            inputs: The inputs for the current step
            num_items_in_batch: Number of items in the current batch
        """
        # Ensure model is on the correct device
        model = model.to(self.device)
        model.train()
        print(f"inputs type: {type(inputs)}, keys: {inputs.keys() if isinstance(inputs, dict) else 'not a dict'}")
        
        # Handle empty inputs - this should not happen if get_train_dataloader is working properly
        if not inputs or not isinstance(inputs, dict) or "pixel_values" not in inputs:
            print(f"Warning: Invalid inputs detected: {inputs}")
            # Create dummy data to prevent crashing - we'll fix the data flow
            x = torch.randn(4, 3, 224, 224).to(self.device)
            labels = torch.randint(0, 100, (4,)).to(self.device)
        else:
            # Ensure inputs are on the correct device
            x = inputs["pixel_values"].to(self.device)
            labels = inputs["labels"].to(self.device)
            
        # Forward pass
        outputs = model(x)
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs, labels)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        # Get gradients and ensure they're on CPU for communication
        gradients = {name: param.grad.cpu() for name, param in model.named_parameters() if param.grad is not None}

        # Send gradients to server and receive averaged gradients
        update, avg_gradients = self.send_recv(gradients)
        
        if not update:
            print(f"Worker {self.worker_id} failed to receive averaged gradients.")
            return loss.detach()

        # Update model parameters with averaged gradients
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in avg_gradients:
                    param.grad = avg_gradients[name].to(self.device)

        return loss.detach()

    def calc_network_latency(self, is_send):
        self.network_latency_list.append(self.end_time - self.start_time)
        if is_send:
            print(f'Send Network latency: {self.end_time - self.start_time}')
        else:
            print(f'Recv Network latency: {self.end_time - self.start_time}')
        self.start_time = 0
        self.end_time = 0

    def print_total_network_latency(self):
        print(f'Total network latency for worker {self.worker_id}: {sum(self.network_latency_list)}')


def custom_collate_fn(batch):
    """
    Custom collate function to properly batch data for our model
    """
    # Print batch structure for debugging
    if len(batch) > 0:
        print(f"collate_fn: batch size={len(batch)}, first item keys={batch[0].keys()}")
    
    # Collate function to preserve the structure of the batch
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }

class Worker:
    def __init__(self, worker_id, host="localhost", port=65432):
        self.worker_id = worker_id
        self.server_host = host
        self.server_port = port
        
        # Set up device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        #     print("Using MPS device")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device")
        
        # Load untrained VGG-13 model
        self.model = models.vgg13(weights=None)
        self.model = self.model.to(self.device)
        print(f"Model moved to {self.device}")

        # Get the absolute path to the distributed-ml-training directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        SPLIT_DIR = os.path.join(project_root, "data", "cifar100_splits")
        
        # Load training dataset split
        train_split_path = os.path.join(SPLIT_DIR, f"train_{self.worker_id}.pth")
        if not os.path.exists(train_split_path):
            raise FileNotFoundError(f"Training dataset split not found at {train_split_path}. Please run prepare_data.py first.")
            
        # Load test dataset split
        test_split_path = os.path.join(SPLIT_DIR, f"test.pth")
        if not os.path.exists(test_split_path):
            raise FileNotFoundError(f"Test dataset split not found at {test_split_path}. Please run prepare_data.py first.")
            
        # Load both dataset splits
        with safe_globals([torch.utils.data.dataset.Subset]):
            train_subset = torch.load(train_split_path, weights_only=False)
            test_subset = torch.load(test_split_path, weights_only=False)
            print(f"Loaded training split containing {len(train_subset)} samples")
            print(f"Loaded test split containing {len(test_subset)} samples")
            
        # Convert the subsets into our custom dataset format
        self.train_dataset = CIFAR100Dataset(train_subset)
        self.eval_dataset = CIFAR100Dataset(test_subset)
        print(f"Created datasets for worker {self.worker_id}")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.eval_dataset)}")

        # # Create DataLoader with custom collate function
        # self.train_dataloader = DataLoader(self.train_dataset, batch_size=4, collate_fn=custom_collate_fn)
        # self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=4, collate_fn=custom_collate_fn)

        # Verify DataLoader output
        for batch in self.train_dataset:
            print(f"train_dataset Batch keys: {batch.keys()}")
            break  # Just to check the first batch

    def compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation
        Args:
            eval_pred: tuple of (predictions, labels)
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}

    def train_worker(self):
        # Initialize the distributed trainer
        self.training_args = TrainingArguments(
            output_dir=f"./results_worker_{self.worker_id}",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            logging_dir=f"./logs_worker_{self.worker_id}",
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            dataloader_pin_memory=True,  # Enable pin_memory for faster data transfer to GPU
        )

        trainer = DistributedTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,  # Add evaluation dataset
            server_host=self.server_host,
            server_port=self.server_port,
            worker_id=self.worker_id,
            compute_metrics=self.compute_metrics,
            device=self.device  # Pass the device to DistributedTrainer
        )

        # Start training
        trainer.train()
        trainer.print_total_network_latency()


def main():
    if len(sys.argv) < 2:
        print("Invalid usage.")
        print("USAGE: python worker_trainer.py <WORKER_ID> [--port PORT]")
        sys.exit(1)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    worker_id = int(sys.argv[1])
    
    # Parse port argument if provided
    port = 60000  # default port
    if len(sys.argv) > 2 and sys.argv[2] == "--port":
        if len(sys.argv) > 3:
            port = int(sys.argv[3])
        else:
            print("Error: --port requires a port number")
            sys.exit(1)
    
    worker = Worker(worker_id, port=port)
    worker.train_worker()


if __name__ == "__main__":
    main()