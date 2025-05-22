import torch

# export VLLM_WORKER_MULTIPROC_METHOD=spawn
torch.multiprocessing.set_start_method("spawn", force=True)

# export LD_LIBRARY_PATH=$HOME/anaconda3/envs/vllm/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
from consts import *
from utils import *
import os
import re
from tqdm import tqdm
from typing import Optional
import tiktoken

import transformers
from prompt_templates import *

delimiters = {
    "section_delimiter": "-",
    "tuple_delimiter": "<|>",
    "completion_delimiter": "<|COMPLETE|>",
    "record_delimiter": "##",
}


class TextChunker:
    def __init__(self, text_max_tokens=600):
        self.text_max_tokens = text_max_tokens

    def _chunk_text(
        self,
        text: str,
        max_tokens: int,
        tokenizer: Optional[tiktoken.Encoding] = None,
        chunk_overlap: int = 100,
    ):
        """Chunk text by token length with overlap."""
        """Text Utilities for LLM. https://github.com/microsoft/graphrag/blob/044516f538788a45728cd1ae6ec55627c0ba5fa7/graphrag/query/llm/text_utils.py"""
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")

        splits: list[str] = []
        input_ids = tokenizer.encode(text)  # type: ignore
        start_idx = 0
        start_chars = []
        cur_idx = min(start_idx + max_tokens, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        while start_idx < len(input_ids):
            splits.append(tokenizer.decode(chunk_ids))
            if start_idx == 0:
                start_chars.append(0)
            else:
                start_chars.append(len(tokenizer.decode(input_ids[:start_idx])))
            start_idx += max_tokens - chunk_overlap
            cur_idx = min(start_idx + max_tokens, len(input_ids))
            chunk_ids = input_ids[start_idx:cur_idx]
        return splits, start_chars

    def get_text_chunks(self, text, file_name=None, save_chunks=False):
        if file_name and os.path.exists(
            f'{PATH["LLM"]["chunked_text"]}/{file_name}.json'
        ):
            chunck_data = load_json_file(
                f'{PATH["LLM"]["chunked_text"]}/{file_name}.json'
            )
            chunks, chunks_start_idxs = [], []
            for span in chunck_data["spans"]:
                chunks.append(chunck_data["text"][span[0] : span[1]])
                chunks_start_idxs.append(span[0])
        else:
            chunks, chunks_start_idxs = self._chunk_text(
                text, max_tokens=self.text_max_tokens, chunk_overlap=100
            )
            if save_chunks and file_name:
                spans = []
                for i, (chunk, start_idx) in enumerate(zip(chunks, chunks_start_idxs)):
                    spans.append((start_idx, start_idx + len(chunk)))
                out = {"text": text, "spans": spans}
                with open(
                    f"../data/chuncked/{self.text_max_tokens}tokens/{file_name}.json",
                    "w",
                ) as file:
                    json.dump(out, file, indent=4)
        return chunks, chunks_start_idxs


class InfoExtractor:
    def __init__(
        self,
        exp="base",
        engine="Llama-3.3-70B-Instruct",
        n_shot=10,
        use_vllm=True,
    ):
        if engine == "NA":
            return
        if exp == "0_shot":
            self.PROMPT_TEMPLATE = PROMPT_TEMPLATE_ZERO_SHOT
        elif exp == "no_rag":
            self.PROMPT_TEMPLATE = PROMPT_TEMPLATE_NO_RAG
        elif exp == "no_relation":
            self.PROMPT_TEMPLATE = PROMPT_TEMPLATE_NO_RELATION
        else:
            print(f"Using base prompt template for exp = {exp}")
            self.PROMPT_TEMPLATE = PROMPT_TEMPLATE
        if "_shot" in exp:
            n_shot = int(exp.split("_")[0])

        self.model = None
        # Load LLM model
        if "Llama" in engine or "climategpt" in engine:
            if "climategpt" in engine:
                model_id = f"eci-io/{engine}"
            elif os.path.exists(f"/data/LLMs/{engine}"):
                model_id = "/data/LLMs/" + engine  # load from local
            else:
                print(f"Downloading from huggingface: {engine}")
                model_id = "meta-llama/Meta-" + engine
            print(f"Loading model from {model_id}")

            if use_vllm:
                from vllm import LLM, SamplingParams

                # https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/generation_config.json
                self.sampling_params = SamplingParams(
                    temperature=0.6,
                    top_p=0.9,
                    stop=[delimiters["completion_delimiter"]],
                    max_tokens=2048,
                )
                self.model = LLM(
                    model=model_id,
                    task="generate",
                    tensor_parallel_size=2,
                    gpu_memory_utilization=0.95,
                    max_model_len=20000,  # 25390,
                    enable_prefix_caching=True,
                )
            else:
                # --------- Huggingface --------
                self.model = transformers.pipeline(
                    "text-generation",
                    model=model_id,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map="auto",
                )
                self.processor = [
                    self.model.tokenizer.eos_token_id,
                    self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]
                self.model.tokenizer.pad_token_id = self.model.tokenizer.eos_token_id

        elif "DeepSeek-V3" in engine:
            use_vllm = False
            from openai import OpenAI
            from credentials import DEEPSEEK

            self.client = OpenAI(api_key=DEEPSEEK, base_url="https://api.deepseek.com")
        else:
            exit(f"Engine {engine} not supported")

        self.use_vllm = use_vllm

        # load n-shot examples
        examples = load_json_file(PATH["LLM"]["examples"])
        self.formatted_examples = ""
        for i, example in enumerate(examples[:n_shot]):
            if exp == "no_rag":
                example = re.sub(
                    r"\nPotential Entities:.*?\n##",
                    "\n##",
                    example,
                    flags=re.DOTALL,
                )
            if exp == "no_relation":
                example = re.sub(
                    r'##\n\("relationship"<\|>.*?\)\n', "", example, flags=re.DOTALL
                )
            self.formatted_examples += f"\nExample {i+1}:\n{example}"

    def parse_response(self, response, with_description=True):
        out = {"entities": [], "relationships": []}
        # trim the response to start from the first entity
        start_index = response.find('("')
        if start_index != 0:
            response = response[start_index:]

        # trim the response to end with <|COMPLETE|>
        # if not self.use_vllm:
        response = response.split(delimiters["completion_delimiter"])[0]

        # split response into records
        response = response.split(delimiters["record_delimiter"])
        response = [
            r.lstrip("\n").rstrip("\n").lstrip("(").rstrip(")") for r in response
        ]

        # split the response into items
        pattern = r"<\s*\|\s*>"
        response = [re.split(pattern, r) for r in response]
        for r in response:
            if "entity" in r[0]:
                if with_description:
                    if len(r) == 4:
                        out["entities"].append(
                            {"name": r[1], "label": r[2], "description": r[3]}
                        )
                elif len(r) == 3:
                    out["entities"].append({"name": r[1], "label": r[2]})

            elif "relationship" in r[0]:
                if len(r) == 4:
                    out["relationships"].append(
                        {"source": r[1], "target": r[2], "relation": r[3]}
                    )
        return out

    def run(
        self,
        text,
        retrieved_nodes,
    ):

        potential_entities = list(retrieved_nodes.keys())
        potential_entities = ", ".join(potential_entities)

        prompt = self.PROMPT_TEMPLATE.format(
            **delimiters,
            formatted_examples=self.formatted_examples,
            input_text=text.replace("{", "").replace("}", ""),
            potential_entities=potential_entities,
        ).format(**delimiters)

        conversation = [{"role": "user", "content": prompt}]
        if self.model:
            if self.use_vllm:
                output = self.model.generate(
                    prompt, sampling_params=self.sampling_params
                )
                pred_content = output[0].outputs[0].text
            else:
                outputs = self.model(
                    conversation,
                    max_new_tokens=4000,
                    pad_token_id=self.model.tokenizer.eos_token_id,
                    # eos_token_id=terminators,
                    # do_sample=True,
                    # temperature=0.6,
                    # top_p=0.9,
                )
                pred_content = outputs[0]["generated_text"][-1]["content"]
        else:
            outputs = self.client.chat.completions.create(
                model="deepseek-chat", messages=conversation, stream=False
            )
            pred_content = outputs.choices[0].message.content
        conversation.append({"role": "assistant", "content": pred_content})
        response = self.parse_response(pred_content)
        return response, conversation


if __name__ == "__main__":
    MODEL = InfoExtractor()
    TEXTCHUNKER = TextChunker()
    full_text = "<heading>ABSTRACT</heading>\nGlobal climate models project significant changes to air temperature and precipitation regimes in many regions of the Northern Hemisphere. These meteorological changes will have associated impacts to surface and shallow subsurface thermal regimes, which are of interest to practitioners and researchers in many disciplines of the natural sciences. For example, groundwater temperature is critical for providing and sustaining suitable thermal habitat for coldwater salmonids. To investigate the surface and subsurface thermal effects of atmospheric climate change, seven downscaled climate scenarios (2046-2065) for a small forested catchment in New Brunswick, Canada were employed to drive the surface energy and moisture flux model, ForHyM2. Results from these seven simulations indicate that climate change-induced increases in air temperature and changes in snow cover could increase summer surface temperatures (range \u22120.30 to +3.49 \u2022 C, mean +1.49 \u2022 C), but decrease winter surface temperatures (range \u22121.12 to +0.08 \u2022 C, mean \u22120.53 \u2022 C) compared to the reference period simulation. Thus, changes to the timing and duration of snow cover will likely decouple changes in mean annual air temperature (mean +2.11 \u2022 C) and mean annual ground surface temperature (mean +1.06 \u2022 C). Projected surface temperature data were then used to drive an empirical surface to groundwater temperature transfer function developed from measured surface and groundwater temperature. Results from the empirical transfer function suggest that changes in groundwater temperature will exhibit seasonality at shallow depths (1.5 m), but be seasonally constant and approximately equivalent to the change in the mean annual surface temperature at deeper depths (8.75 m). The simulated increases in future groundwater temperature suggest that the thermal sensitivity of baseflow-dominated streams to decadal climate change may be greater than previous studies have indicated.\n<heading>Introduction</heading>\n\n<heading>Drivers and importance of ground surface temperature</heading>\nThe impact of climate change on ground surface temperature (GST) is of interest to a diversity of scientific disciplines. For example, hydrologists are concerned with the influence of surface freezing and thawing on infiltration and runoff rates (Williams and Smith, 1989), agricultural scientists have shown that seed germination is affected by surface and near-surface temperature (Mondoni et al., 2012), and geotechnical engineers have linked soil strength properties to surface/subsurface temperature (Andersland and Ladanyi, 1994). Increased GST could also enhance decay rates and CO 2 release from soils and thereby act as a positive feedback mechanism to climate change (Eliasson et al., 2005). Potential effects of changes in winter GST include: altered nutrient concentrations in soil water, enhanced winter root mortality, and decreased runoff quality (Mellander et al., 2007).\n<heading>B. L. Kurylyk et al.: Surface and groundwater temperature response to climate change</heading>\nIncreases in mean annual air"
    file_name = None
    text_chunks = TEXTCHUNKER.get_text_chunks(full_text)
    pbar = tqdm(list(zip(*text_chunks)))
    pbar.set_description(f"LLM running on File [{file_name}]'s chunks")

    for text, start_idx in pbar:
        retrieved_nodes = {
            "soil strength": ["project"],
            "ground surface temperature": ["variable"],
            "subsurface temperature": ["variable"],
            "surface and near-surface temperature": ["variable"],
            "snow cover": ["variable"],
            "decadal climate change": ["variable"],
            "regions": ["location"],
            "soil water": ["variable"],
            "soils": ["variable"],
            "baseflow-dominated streams": ["location"],
            "seasonality": ["variable"],
            "mean annual surface temperature": ["variable"],
            "New Brunswick": ["location"],
            "Northern Hemisphere": ["location"],
            "infiltration and runoff rates": ["variable"],
            "Projected surface temperature data": ["variable"],
            "mean annual air temperature": ["variable"],
            "air temperature": ["variable"],
        }
        output, conversation = MODEL.run(text, retrieved_nodes)
        print(conversation)
        print("output:", output)
