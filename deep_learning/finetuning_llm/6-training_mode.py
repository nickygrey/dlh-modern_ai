#!/usr/bin/env python3
"""Module to train, evaluate, and save a Hugging Face model."""
import transformers


def train_model(model, training_args, train_dataset, eval_dataset,
                test_dataset, tokenizer, data_collator, compute_metrics,
                model_save_name):
    """Train, evaluate, and save a pre-trained language model.

    Args:
        model (transformers.PreTrainedModel): Pre-trained model to fine-tune.
        training_args (transformers.TrainingArguments): Training arguments with
            hyperparameters and strategies.
        train_dataset (datasets.Dataset): Tokenized training dataset.
        eval_dataset (datasets.Dataset): Tokenized validation dataset.
        test_dataset (datasets.Dataset): Tokenized test dataset.
        tokenizer (transformers.PreTrainedTokenizerBase): Tokenizer or
            processing class used for preprocessing.
        data_collator (callable): Data collator for dynamic padding.
        compute_metrics (callable): Function computing evaluation metrics from
            predictions and labels.
        model_save_name (str): Directory to save the trained model and
            tokenizer.

    Returns:
        tuple: (trainer, train_results, test_results) where:
            trainer (transformers.Trainer): Hugging Face Trainer instance.
            train_results: Training metrics and logs.
            test_results (dict): Evaluation results on the test dataset.
    """
    try:
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
    except TypeError:
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

    train_results = trainer.train()
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    trainer.save_model(model_save_name)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(model_save_name)

    if getattr(training_args, "push_to_hub", False):
        try:
            trainer.push_to_hub()
        except Exception:
            pass

    return trainer, train_results, test_results
