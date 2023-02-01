import pathlib
import modal
from pytorch_lightning.utilities.seed import seed_everything
import pytorch_lightning as pl
from pl_bolts.datamodules import MNISTDataModule
from modal_fs_testing.model import Encoder, LitAutoEncoder, Decoder

stub = modal.Stub(
    "fs-infer-model-pl",
    image=modal.Image.conda()
    .conda_install(
        "cudatoolkit=11.2",
        "cudnn=8.1.0",
        "cuda-nvcc",
        channels=["conda-forge", "nvidia"],
    )
    .pip_install("protobuf==3.20.0")
    .pip_install("pytorch-lightning==1.6.0")
    .pip_install("torchvision")
    .pip_install("lightning-bolts==0.4.0"),
)

volume = modal.SharedVolume().persist("model_checkpoints")
ROOT = "/root"
MODEL_CACHE = pathlib.Path(ROOT, "models")


@stub.function(shared_volumes={str(MODEL_CACHE): volume}, gpu="any", timeout=600)
def infer_model(model_name: str):

    # Setup
    seed_everything(42, workers=True)
    dm = MNISTDataModule(".", batch_size=256, num_workers=24)
    MODEL_EXPORT_PATH = pathlib.Path(MODEL_CACHE, f"{model_name}.ckpt")
    MODEL_EXPORT_PATH.parent.mkdir(exist_ok=True)

    # Load and test model
    autoencoder = LitAutoEncoder(Encoder(), Decoder())
    model = autoencoder.load_from_checkpoint(checkpoint_path=MODEL_EXPORT_PATH)
    trainer = pl.Trainer(devices="auto", accelerator="auto", max_epochs=1)
    results = trainer.test(model=model, datamodule=dm)
    return results


@stub.local_entrypoint
def main():
    print("Results: ", infer_model.call("model-a"))
