import mlflow
import mlflow.pytorch as ml_pytorch
import torch

class DDPExperiment:
    def __init__(self, model: torch.nn.Module, experiment_name=None, run_name=None, rank=0):
        self.model = model
        self.rank = rank
        self.experiment_name = experiment_name
        self.run_name = run_name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        self.mlflow_context = mlflow.start_run(run_name=run_name, experiment_id=(experiment.experiment_id if experiment is not None else None))

    def __enter__(self):
        self.mlflow_context.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.rank == 0:
            scripted_model = torch.jit.script(self.model)
            scripted_emedding = torch.jit.script(self.model.embedding)
            ml_pytorch.log_model(self.model, "model")
            ml_pytorch.log_model(scripted_model, "scripted_model")
            ml_pytorch.log_model(self.model.embedding, "embedding")
            ml_pytorch.log_model(scripted_emedding, "scripted_embedding")
        self.mlflow_context.__exit__(exc_type, exc_val, exc_tb)
