import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

from train import get_ds, greedy_decode
from config import get_config
cfg = get_config()

from model import build_transformer

class Transformerpyl(pl.LightningModule):
    def __init__(self, num_examples):
        super(Transformerpyl, self).__init__()
        self.learning_rate = cfg["lr"]
        self.training_dataloader, self.valid_dataloader, self.tokenizer_src, self.tokenizer_tgt = get_ds(cfg)
        model = build_transformer(
              self.tokenizer_src.get_vocab_size(),
              self.tokenizer_tgt.get_vocab_size(),
              cfg["seq_len"],
              cfg["seq_len"],
              d_model=cfg["d_model"],
        )
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1)
        self.max_len = cfg["seq_len"]
        self.devicegpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_examples = num_examples   

    def training_step(self, batch, batch_idx):
        encoder_input = batch['encoder_input']
        decoder_input = batch['decoder_input']
        encoder_mask = batch['encoder_mask']
        decoder_mask = batch['decoder_mask']

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = self.model.encode(
            encoder_input, encoder_mask
        )  # (B, seq_len, d_model)
        decoder_output = self.model.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )  #
        proj_output = self.model.project(decoder_output)

        # Compare the output with the label
        label = batch["label"].to(self.devicegpu)  # (B, seq_len)

        # Compute the loss using a simple cross entropy
        loss = self.loss_fn(
            proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1)
        )

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        source_texts = []
        expected = []
        predicted = [] 
        with torch.no_grad():
          encoder_input = batch['encoder_input']
          encoder_mask = batch['encoder_mask']

          assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

          model_out = greedy_decode(self.model, encoder_input, encoder_mask, self.tokenizer_src, self.tokenizer_tgt, self.max_len, self.devicegpu)
          source_text = batch["src_text"][0]
          target_text = batch["tgt_text"][0]
          model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

          source_texts.append(source_text)
          expected.append(target_text)
          predicted.append(model_out_text)

          # Print the source, target, and model output
          print("-"*10)
          print(f"{f'SOURCE: ':>12}{source_text}")
          print(f"{f'TARGET: ':>12}{target_text}")
          print(f"{f'PREDICTED: ':>12}{model_out_text}")

        metric = torchmetrics.text.CharErrorRate()
        cer = metric(predicted, expected)
        self.log("cer", cer)

        metric = torchmetrics.text.WordErrorRate()
        wer = metric(predicted, expected)
        self.log("wer", wer)   

        metric = torchmetrics.text.BLEUScore()
        bleu = metric(predicted, expected) 
        self.log("bleu", bleu)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-9)
        return optimizer
    
    def train_dataloader(self):
        return self.training_dataloader


    def val_dataloader(self):
        return self.valid_dataloader