import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import socket
import pickle
import struct
import time
from my_datasets import CIFAR10Dataset
from distributed_trainer import DistributedTrainer
from transformers import Trainer, TrainingArguments
from torchvision import transforms
import torchvision.models as models
import random
import numpy as np
import os
from sklearn.metrics import accuracy_score
from torch.serialization import safe_globals

train_args = TrainingArguments(
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            learning_rate=0.001,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            dataloader_pin_memory=True,  # Enable pin_memory for faster data transfer to GPU
            report_to="none",
            logging_first_step=True,  # Log metrics for the first step
        )

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
        
        # Load untrained EfficientNetB0 model
        self.model = models.efficientnet_b0(weights=None)
        # Modify the model's classifier to output 10 classes (CIFAR10)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 10)
        self.model = self.model.to(self.device)
        print(f"Model moved to {self.device}")

        # Get the absolute path to the distributed-ml-training directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        SPLIT_DIR = os.path.join(project_root, "data", "cifar10_splits")
        
        # Load training dataset split
        train_split_path = os.path.join(SPLIT_DIR, f"train_{self.worker_id}.pth")
        if not os.path.exists(train_split_path):
            raise FileNotFoundError(f"Training dataset split not found at {train_split_path}. Please run prepare_data.py first to generate CIFAR10 splits.")
            
        # Load test dataset split
        test_split_path = os.path.join(SPLIT_DIR, f"test.pth")
        if not os.path.exists(test_split_path):
            raise FileNotFoundError(f"Test dataset split not found at {test_split_path}. Please run prepare_data.py first to generate CIFAR10 splits.")
            
        # Load both dataset splits
        with safe_globals([torch.utils.data.dataset.Subset]):
            train_subset = torch.load(train_split_path, weights_only=False)
            test_subset = torch.load(test_split_path, weights_only=False)
            print(f"Loaded CIFAR10 training split containing {len(train_subset)} samples")
            print(f"Loaded CIFAR10 test split containing {len(test_subset)} samples")
            
        # Convert the subsets into our custom dataset format
        self.train_dataset = CIFAR10Dataset(train_subset)
        self.eval_dataset = CIFAR10Dataset(test_subset)
        print(f"Created CIFAR10 datasets for worker {self.worker_id}")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.eval_dataset)}")

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
        self.training_args = train_args
        self.training_args.output_dir = f"./results_worker_{self.worker_id}"
        self.training_args.logging_dir = f"./logs_worker_{self.worker_id}"

        trainer = DistributedTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,  # Make sure this is passed
            server_host=self.server_host,
            server_port=self.server_port,
            worker_id=self.worker_id,
            device=self.device
        )

        # Start training
        print(f"Worker {self.worker_id} starting training with evaluation...")
        train_result = trainer.train()
        print(f"Worker {self.worker_id} training completed. Results: {train_result}")
        
        # Explicitly evaluate after training
        eval_results = trainer.evaluate()
        print(f"Worker {self.worker_id} final evaluation results: {eval_results}")
        
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