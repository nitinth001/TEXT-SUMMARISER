from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from textSummarizer.entity import ModelTrainerConfig
from datasets import load_from_disk
import torch
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)

        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        # Load dataset
        dataset = load_from_disk(self.config.data_path)

        # Training arguments
        trainer_args = TrainingArguments(
          output_dir=self.config.root_dir,
         num_train_epochs=int(self.config.num_train_epochs),
         warmup_steps=int(self.config.warmup_steps),
    per_device_train_batch_size=int(self.config.per_device_train_batch_size),
    per_device_eval_batch_size=int(self.config.per_device_train_batch_size),
    weight_decay=float(self.config.weight_decay),
    logging_steps=int(self.config.logging_steps),

    evaluation_strategy=self.config.eval_strategy,   # ✅ correct name
    eval_steps=int(self.config.eval_steps),           # ✅ FORCE INT

    save_steps=int(float(self.config.save_steps)),    # ✅ handle 1e6
    gradient_accumulation_steps=int(self.config.gradient_accumulation_steps)
)

        # Trainer
        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            data_collator=data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"]
        )

        # Train
        trainer.train()

        # Save model
        model_pegasus.save_pretrained(
            os.path.join(self.config.root_dir, "pegasus-samsum-model")
        )
        tokenizer.save_pretrained(
            os.path.join(self.config.root_dir, "tokenizer")
        )