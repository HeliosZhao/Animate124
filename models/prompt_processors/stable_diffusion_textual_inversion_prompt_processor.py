import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline
import threestudio

from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt

@threestudio.register("animate124-textual-inversion-prompt-processor")
class SDTextualInversionPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        ## add cn prompt and ti path
        ## NOTE When using dataclass, the data type must be mentioned.
        learned_embeds_path: str = ""
        use_cache: bool = False # this should be false, since the textual inversion may change while the cache name not 
        spawn: bool = False

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        cleanup()

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B 77 768"], Float[Tensor, "B 77 768"]]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        # Tokenize text and get embeddings
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
            uncond_text_embeddings = self.text_encoder(
                uncond_tokens.input_ids.to(self.device)
            )[0]

        return text_embeddings, uncond_text_embeddings

    # @staticmethod
    def spawn_func(self, pretrained_model_name_or_path, prompts, cache_dir):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)
        if self.cfg.learned_embeds_path:
            pipe.load_textual_inversion(self.cfg.learned_embeds_path)
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder

        token_state_dict = torch.load(self.cfg.learned_embeds_path)
        token_name, _ = next(iter(token_state_dict.items()))

        prompts = [p.replace("<token>", token_name) for p in prompts]
        threestudio.info(
            f"After textual inversion token replacing: [{prompts}]"
        )
        

        with torch.no_grad():
            tokens = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_embeddings = text_encoder(tokens.input_ids.to(text_encoder.device))[0]

        for prompt, embedding in zip(prompts, text_embeddings):
            save_prompt = prompt.replace(token_name, "<token>") # back to token for saving
            torch.save(
                embedding,
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, save_prompt)}.pt",
                ),
            )

        del text_encoder, tokenizer, pipe
