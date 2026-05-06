from pathlib import Path
from typing import Optional

import mlflow
import torch
from google.cloud import storage


class CheckpointManager:
    def __init__(
        self,
        save_dir: str,
        GCS_bucket: Optional[str] = None,
        experiment_name: str = "llm-training",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.GCS_bucket = GCS_bucket
        self.experiment_name = experiment_name

        if GCS_bucket:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(GCS_bucket.replace("gs://", ""))

    def save(
        self,
        model: torch.nn.Module,
        optimizer_state: dict,
        step: int,
        loss: float,
        metrics: Optional[dict] = None,
    ) -> Path:
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer_state,
            "step": step,
            "loss": loss,
            "metrics": metrics or {},
        }

        path = self.save_dir / f"checkpoint_{step}.pt"
        torch.save(checkpoint, path)

        if self.GCS_bucket:
            self._upload_to_gcs(path)

        return path

    def save_best(
        self,
        model: torch.nn.Module,
        optimizer_state: dict,
        step: int,
        loss: float,
    ) -> Path:
        path = self.save_dir / "best.pt"
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer_state,
            "step": step,
            "loss": loss,
        }
        torch.save(checkpoint, path)

        if self.GCS_bucket:
            self._upload_to_gcs(path)

        return path

    def load(self, path: str | Path) -> dict:
        path = Path(path)
        if not path.exists() and self.GCS_bucket:
            self._download_from_gcs(path)

        return torch.load(path, map_location="cpu")

    def _upload_to_gcs(self, path: Path) -> None:
        blob = self.bucket.blob(str(path))
        blob.upload_from_filename(str(path))

    def _download_from_gcs(self, path: Path) -> None:
        blob = self.bucket.blob(str(path))
        blob.download_to_filename(str(path))


class MLflowLogger:
    def __init__(
        self,
        experiment_name: str = "llm-training",
        tracking_uri: Optional[str] = None,
    ):
        mlflow.set_experiment(experiment_name)
        self.tracking_uri = tracking_uri

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str | Path) -> None:
        mlflow.log_artifact(str(local_path))
