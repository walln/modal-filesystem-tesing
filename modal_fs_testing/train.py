import pathlib
import modal
from pytorch_lightning.utilities.seed import seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pl_bolts.datamodules import MNISTDataModule


from modal_fs_testing.model import Encoder, LitAutoEncoder, Decoder

stub = modal.Stub(
    "fs-test-train-model-pl",
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
def train_model(model_name: str):

    # Setup
    seed_everything(42, workers=True)
    dm = MNISTDataModule(".", batch_size=256, num_workers=24)
    early_stopping = EarlyStopping("val_loss")
    MODEL_EXPORT_PATH = pathlib.Path(MODEL_CACHE, f"{model_name}.ckpt")
    MODEL_EXPORT_PATH.parent.mkdir(exist_ok=True)

    # Create and train model
    autoencoder = LitAutoEncoder(Encoder(), Decoder())
    trainer = pl.Trainer(
        callbacks=[early_stopping], devices="auto", accelerator="auto", max_epochs=1
    )
    trainer.fit(model=autoencoder, datamodule=dm)

    # Save model checkpoint
    trainer.save_checkpoint(MODEL_EXPORT_PATH)

    return True


@stub.local_entrypoint
def main():
    print("Success: ", train_model.call("model-b"))
