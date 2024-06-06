import abc

import mlflow
from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.join import Join
import torch.multiprocessing as mp
import torch.distributed as dist
import os
from experiment import DDPExperiment
import torchmetrics
from cli_args import TrainArguments


def run(rank, fn, args: TrainArguments):
    setup(rank, args)
    fn(rank, args)
    teardown()


def setup(rank, args: TrainArguments):
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=args.number_devices)


def teardown():
    dist.destroy_process_group()


class DDPRunner():
    def __init__(self, fn, args: TrainArguments):
        self.fn = fn
        self.args = args

    def run(self):
        mp.spawn(run, nprocs=self.args.number_devices, args=(self.fn, self.args), join=True)


# from torchmetrics import Accuracy
class Model(torch.nn.Module):
    @abc.abstractmethod
    def get_optimizer(self) -> torch.optim.Optimizer:
        pass

    @abc.abstractmethod
    def get_loss_function(self) -> torch.nn.modules.loss._Loss:
        pass

    @abc.abstractmethod
    def get_learning_rate_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        pass


class DDPTrainer:
    def __init__(self, experiment: DDPExperiment, epochs=1, device="cpu"):
        self.epochs = epochs
        self.device = device
        self.experiment = experiment
        self.model = experiment.model
        self.gradient_scaler = torch.cuda.amp.GradScaler()
        self.optimizer = self.model.get_optimizer()
        self.loss_function = self.model.get_loss_function()
        self.lr_scheduler = self.model.get_learning_rate_scheduler()
        self.ddp_model = DDP(self.model, device_ids=[experiment.rank])
        self.train_loss_mean = torchmetrics.aggregation.MeanMetric().to(self.experiment.rank)
        self.train_accuracy_mean = torchmetrics.aggregation.MeanMetric().to(self.experiment.rank)
        self.train_accuracy = torchmetrics.classification.MulticlassAccuracy(
            num_classes=self.model.vocabulary.__len__()).to(
            self.experiment.rank)
        self.validation_accuracy = torchmetrics.classification.MulticlassAccuracy(
            num_classes=self.model.vocabulary.__len__()).to(
            self.experiment.rank)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def train_batches(self, epoch, batches):
        self.train_loss_mean.reset()
        self.train_accuracy_mean.reset()

        for batch_idx, batch in enumerate(batches):
            self.optimizer.zero_grad(set_to_none=True)
            input, targets = batch
            input = input.to(device=self.device)
            targets = targets.to(device=self.device)
            # lengths should remain on cpu as all processing what needs lengths must be done on cpu
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                output = self.ddp_model(input)
                loss = self.loss_function(output, targets)

            self.gradient_scaler.scale(loss).backward()
            self.gradient_scaler.step(self.optimizer)
            self.gradient_scaler.update()

            # subsample metrics for sense of progress
            if batch_idx % 100 == 0:
                _, predicted = torch.max(output, 1)
                correct = (predicted == targets).float().sum().item()
                loss = self.train_loss_mean(loss.item()).item()
                accuracy = self.train_accuracy_mean(correct / targets.size(0)).item()
                mlflow.log_metrics({
                    f"loss_train_{epoch + 1}": loss,
                    f"accuracy_train_{epoch + 1}": accuracy
                }, step=batch_idx)
                batches.set_postfix(batch_loss=loss, batch_accuracy=accuracy)

        # self.debug("entering barrier")
        # dist.barrier()
        # self.debug("exiting barrier")
        # # metrics = torch.tensor([sum(losses)/len(losses), sum(accuracies)/len(accuracies)], device=self.experiment.rank)
        # # self.debug("reducing loss and accuracy")
        # # dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        # # self.debug("all reduced")
        # mlflow.log_metrics({
        #     "loss_train": self.train_loss_mean.compute().item(),
        #     # "accuracy_train": self.train_accuracy_mean.compute().item()
        # }, step=epoch+1)

    def train_epochs(self, epochs, train_dataloader):
        for epoch in range(epochs):
            self.train_epoch(epoch, train_dataloader)

    def train_epoch(self, epoch, train_dataloader):
        with tqdm(train_dataloader, unit="batch", disable=(self.experiment.rank > 0 or os.environ.get("TQDM_DISABLE") is not None)) as batches_iterator:
            batches_iterator.set_description(f"Epoch {epoch} train; rank {self.experiment.rank}")
            self.ddp_model.train()
            self.train_batches(epoch, batches_iterator)
            self.lr_scheduler.step()
            self.train_accuracy.reset()

    def train(self, train_dataloader):
        with Join([self.ddp_model]):
            self.train_epochs(self.epochs, train_dataloader)

    def __validate(self, epoch, validate_dataloader):
        total_loss = 0.0
        average_loss = 0.0
        total_accuracy = 0.0
        average_accuracy = 0.0

        with (tqdm(validate_dataloader, unit="batch", disable=(self.experiment.rank > 0 or os.environ.get("TQDM_DISABLE") is not None)) as batches_iterator):
            batches_iterator.set_description(f"Epoch {epoch} validate; rank {self.experiment.rank}")

            self.ddp_model.module.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(batches_iterator):
                    input, targets = batch
                    input = input.to(device=self.device)
                    targets = targets.to(device=self.device)
                    # lengths should remain on cpu as all processing what needs lengths must be done on cpu
                    with torch.autocast(device_type=self.device, dtype=torch.float16):
                        output = self.ddp_model.module(input)
                        loss = self.loss_function(output, targets)

                    # if batch_idx % 100 == 0:
                    _, predicted = torch.max(output, 1)
                    correct = (predicted == targets).float().sum().item()
                    total_accuracy += correct / targets.size(0)
                    # mlflow.log_metric("loss_validate", loss.item(), step=batch_idx)

                    total_loss += loss.item()
                    # batches_iterator.set_postfix(batch_loss=loss.item(), batch_accuracy=accuracy)
                    batches_iterator.set_postfix(batch_loss=loss.item())
            average_loss = total_loss / (batch_idx + 1.0)
            average_accuracy = total_accuracy / (batch_idx + 1.0)
            mlflow.log_metric("loss_validate", average_loss, step=epoch + 1)
            mlflow.log_metric("accuracy_validate", average_accuracy, step=epoch + 1)

            dist.barrier()
            metrics = torch.tensor([average_loss, average_accuracy], device=self.experiment.rank)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM, async_op=False)
            mlflow.log_metric("loss_validate", metrics[0] / dist.get_world_size(), step=epoch + 1)
            mlflow.log_metric("accuracy_validate", metrics[1] / dist.get_world_size(), step=epoch + 1)

    def validate(self, validate_dataloader):
        with Join([self.ddp_model]):
            self.__validate(0, validate_dataloader)

    def debug(self, message):
        print(f"rank: {self.experiment.rank}; {message}")

    def fit(self, train_dataloader, validate_dataloader):
        for epoch in range(self.epochs):
            with Join([self.ddp_model]):
                self.train_epoch(epoch, train_dataloader)
            with Join([self.ddp_model]):
                self.__validate(epoch, validate_dataloader)
            dist.barrier()
