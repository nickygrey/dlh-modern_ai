#!/usr/bin/env python3
"""Module to create a translation pipeline using Transformers."""
import transformers


def translate_text(model_name, src_lang=None, tgt_lang=None):
    """Create a translation pipeline for language translation.

    Args:
        model_name (str): Name of the pre-trained model to use.
        src_lang (str, optional): Source language code (e.g., 'en').
            Defaults to None.
        tgt_lang (str, optional): Target language code (e.g., 'fr').
            Defaults to None.

    Returns:
        transformers.pipelines.Pipeline: Hugging Face translation pipeline.
    """
    kwargs = {}
    if src_lang is not None:
        kwargs["src_lang"] = src_lang
    if tgt_lang is not None:
        kwargs["tgt_lang"] = tgt_lang

    return transformers.pipeline(
        task="translation",
        model=model_name,
        **kwargs
    )
