from datetime import datetime
import os
from fastapi import FastAPI
import pathlib
import modal
from pydantic import BaseModel
from pytorch_lightning.utilities.seed import seed_everything
import pytorch_lightning as pl
from pl_bolts.datamodules import MNISTDataModule

from modal_fs_testing.model import Encoder, LitAutoEncoder, Decoder


web_app = FastAPI()

stub = modal.Stub("fs-server-pl")
image = (
    modal.Image.conda()
    .conda_install(
        "cudatoolkit=11.2",
        "cudnn=8.1.0",
        "cuda-nvcc",
        channels=["conda-forge", "nvidia"],
    )
    .pip_install("protobuf==3.20.0")
    .pip_install("pytorch-lightning==1.6.0")
    .pip_install("torchvision")
    .pip_install("lightning-bolts==0.4.0")
)


volume = modal.SharedVolume().persist("model_checkpoints")
ROOT = "/root"
MODEL_CACHE = pathlib.Path(ROOT, "models")


class RequestBody(BaseModel):
    model_name: str = "model-a"


@web_app.post("/infer")
async def bar(request: RequestBody):
    seed_everything(42, workers=True)
    dm = MNISTDataModule(".", batch_size=256, num_workers=24)
    MODEL_EXPORT_PATH = pathlib.Path(MODEL_CACHE, f"{request.model_name}.ckpt")
    MODEL_EXPORT_PATH.parent.mkdir(exist_ok=True)

    # Check if the model exists
    if not os.path.exists(MODEL_EXPORT_PATH):
        return {"error": "No such model exists"}

    file_size = os.path.getsize(MODEL_EXPORT_PATH)

    # Load and test model
    autoencoder = LitAutoEncoder(Encoder(), Decoder())

    start_time = datetime.now()
    model = autoencoder.load_from_checkpoint(checkpoint_path=MODEL_EXPORT_PATH)
    end_time = datetime.now()

    delta = end_time - start_time

    trainer = pl.Trainer(devices="auto", accelerator="auto", max_epochs=1)
    results = trainer.test(model=model, datamodule=dm)

    return {
        "time_difference_s": delta.total_seconds(),
        "time_dfference_ms": delta.total_seconds() * 1000,
        "start_time": start_time,
        "end_time": end_time,
        "results": str(results),
        "model_size": file_size / 1_000_000,
    }


@stub.asgi(
    image=image, shared_volumes={str(MODEL_CACHE): volume}, gpu="any", timeout=600
)
def fastapi_app():
    return web_app
