from typing import List

import numpy as np

import transformers


class Speech2Text:
    """Minimalist module for converting speech to text using fairseqS2T."""

    def __init__(self) -> None:
        self.model = (transformers.Speech2TextForConditionalGeneration
                      .from_pretrained("facebook/s2t-small-librispeech-asr"))
        self.processor = (
            transformers.Speech2TextProcessor.from_pretrained(
                "facebook/s2t-small-librispeech-asr")
        )

    def __call__(self,
                 audio: np.ndarray,
                 sample_rate: int = 16000) -> str:
        """

        Args:
            audio (np.ndarray): Audio as a 1d np.ndarray.
            sample_rate (int): Sampling rate of the audio. Defaults to 16000 Hz.

        Returns:
            List[str]: Transcript for the audio.
        """
        inputs = self.processor(audio,
                                sampling_rate=sample_rate,
                                return_tensors="pt")
        generated_ids = self.model.generate(
            inputs["input_features"],
            attention_mask=inputs["attention_mask"]
        )

        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0]


if __name__ == "__main__":
    import datasets # pylint: disable=import-errors

    s2t = Speech2Text()
    ds = datasets.load_dataset(
        "hf-internal-testing/librispeech_asr_demo",
        "clean",
        split="validation"
    )
    print(s2t(ds[0]["audio"]["sampling_rate"], ds[0]["audio"]["sampling_rate"]))
