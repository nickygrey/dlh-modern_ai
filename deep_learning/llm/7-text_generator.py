#!/usr/bin/env python3
"""Module to create a text generation pipeline with custom parameters."""
import transformers


def create_text_generator(model_name, prompt, max_new_tokens,
                          temperature, repetition_penalty,
                          no_repeat_ngram_size):
    """Create a text generation pipeline and generate text from a prompt.

    Args:
        model_name (str): Name of the pre-trained model to use.
        prompt (str): The input text to start generation from.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Controls randomness of generation.
        repetition_penalty (float): Penalizes repeated tokens.
        no_repeat_ngram_size (int): Prevents repeating n-word sequences.

    Returns:
        tuple: (generator, output) where:
            generator (transformers.pipelines.Pipeline): Text generation
                pipeline object.
            output (list[dict]): List of generated text predictions.
    """
    generator = transformers.pipeline(
        task="text-generation",
        model=model_name
    )
    generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id

    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=generator.tokenizer.eos_token_id,
        do_sample=True
    )

    return generator, output
