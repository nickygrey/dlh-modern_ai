#!/usr/bin/env python3
"""Module to generate image captions using a BLIP Vision-Language Model."""
import PIL.Image
import transformers


def image_captioner(model, image_path, max_new_tokens):
    """Generate a textual description of an image using a BLIP model.

    Args:
        model (str): Name of the pre-trained image captioning model to use.
        image_path (str): Path to the image file to caption.
        max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: Generated textual description of the image.
    """
    processor = transformers.BlipProcessor.from_pretrained(model)
    blip_model = (
        transformers.BlipForConditionalGeneration.from_pretrained(model)
    )

    image = PIL.Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    tokens = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(tokens[0], skip_special_tokens=True).strip()

    return caption
