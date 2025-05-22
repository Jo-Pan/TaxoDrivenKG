from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm
from collections import defaultdict
from utils import *
from consts import *
import numpy as np
import copy

np.bool = np.bool_

import torch
import spacy
import os
from transformers import Trainer as TransformersTrainer


class retriever:
    def __init__(self, init_model=True, init_prev_retrieved=True):
        print("Initializing retriever...")
        if init_model:
            llm = Ollama(model="llama3.3", request_timeout=360.0)
            Settings.llm = llm
            # Settings.embed_model = HuggingFaceEmbedding(model_name="WhereIsAI/UAE-Large-V1")
            # self.threshold = 0.7

            Settings.embed_model = HuggingFaceEmbedding(
                model_name="nvidia/NV-Embed-v2", trust_remote_code=True
            )
            self.threshold = 0.5
            storage_context = StorageContext.from_defaults(
                persist_dir=PATH["RAG"]["vector_index"]
            )
            vector_index = load_index_from_storage(storage_context)
            self.retriever = vector_index.as_retriever(similarity_top_k=5)
        self.nlp = spacy.load("en_core_web_sm")
        self.GCMD = load_json_file(PATH["GCMD"])
        print("Loading previous retrieved...")
        if init_prev_retrieved and os.path.exists(PATH["RAG"]["prev_retrieved"]):
            self.RETRIEVED = load_json_file(PATH["RAG"]["prev_retrieved"])
        else:
            self.RETRIEVED = {}
        self.skip_words = [
            "the",
            "this",
            "these",
            "those",
            "that",
            "one",
            "two",
            "a",
            "an",
            "his",
            "her",
            "many",
            "they",
            "their",
            "et al",
            "some",
            "such",
            "its",
            "all",
            "each",
            "both",
            "our",
            "we",
            "us",
            "you",
            "your",
            "my",
            "mine",
            "he",
            "she",
            "it",
            "I",
            "me",
            "who",
            "whom",
            "whose",
            "which",
            "what",
            "any",
            "few",
            "several",
            "more",
            "most",
            "other",
            "another",
            "much",
            "him",
            "hers",
            "ours",
            "yours",
            "theirs",
            "there",
            "here",
            "and",
            "or",
            "nor",
            "but",
            "if",
            "while",
            "because",
            "to",
            "from",
            "of",
            "for",
            "with",
            "on",
            "at",
            "by",
            "about",
            "as",
            "in",
            "into",
            "upon",
            "like",
            "through",
            "over",
            "under",
            "between",
            "since",
            "until",
            "dataset",
        ]

        self.skip_chars = [
            ".",
            "/",
            "%",
            "(",
            "fig",
            "table",
        ]
        self.skip_phrases = set(LABELS_DICT["label_mapper"].keys())
        self.skip_phrases.add("dataset")
        print("Retriever initialized.")

    def extract_noun_phrases(self, sentence):
        doc = self.nlp(sentence)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        return noun_phrases

    def filter_noun_phrases(self, noun_phrases):
        noun_phrases2 = set()
        for p in noun_phrases:
            p = p.replace("\n", "")
            skip = False
            # skip if p contains unicode string
            if any(ord(char) > 128 for char in p):
                continue

            while len(p) > 1:
                # remove starting non-characters and non-digits
                if not p[0].isalpha() and not p[0].isdigit():
                    p = p[1:]
                # remove ending non-characters and non-digits
                elif not p[-1].isalpha() and not p[-1].isdigit():
                    p = p[:-1]
                else:
                    break
            if len(p) <= 3:
                continue
            for w in self.skip_words:
                if p.lower().startswith(w + " "):
                    p = p[len(w) + 1 :]
            for c in self.skip_chars:
                if c in p.lower():
                    skip = True
                    break
            if (
                skip
                or p.lower()
                in [
                    "models",
                    "number",
                    "study",
                    "phase",
                    "people" "need",
                    "what",
                    "which",
                    "year",
                    "a",
                    "some",
                    "many",
                ]
                + self.skip_words
            ):
                continue
            if len(p) > 3:
                noun_phrases2.add(p)
        return noun_phrases2

    def get_label(self, tags):
        for t in tags:
            if t in LABELS_DICT["label_mapper"]:
                return LABELS_DICT["label_mapper"][t]
        return "variable"

    def run(self, text, tqdm_disable=False, main_pbar=None):
        # Get noun phrases
        noun_phrases = self.extract_noun_phrases(text)
        noun_phrases2 = self.filter_noun_phrases(noun_phrases)

        # Initialize output
        row = {}
        pbar = tqdm(noun_phrases2, disable=tqdm_disable)
        pbar.set_description("Retrieving noun phrases")
        phrase_i = 0
        for phrase in pbar:
            phrase_i += 1
            if main_pbar:
                main_pbar.set_description(
                    f"In doc, Retrieving noun phrases - [{phrase}]: [{phrase_i}/{len(noun_phrases2)}] Total Retrieved: {len(self.RETRIEVED):,}"
                )
            if phrase in self.skip_phrases:
                continue
            # if the phrase have been retrieved
            if phrase in self.RETRIEVED:
                if self.RETRIEVED[phrase] != []:
                    row[phrase] = self.RETRIEVED[phrase]
            else:
                # retrieve with the retriever
                node = self.retriever.retrieve(phrase)[0]

                # filter results
                if node.score > self.threshold:
                    uuid = node.metadata["uuid"]
                    label = self.get_label(self.GCMD[uuid]["tags"])
                    row[phrase] = [
                        label,
                        uuid,
                        self.GCMD[uuid]["prefLabel"],
                        node.score,
                    ]
                    self.RETRIEVED[phrase] = row[phrase]
                else:
                    self.RETRIEVED[phrase] = []
        return row

    def retrieve_by_def(self, entity_name, entity_def):
        text = f"Name: {entity_name}\nDefinition: {entity_def}"
        node = self.retriever.retrieve(text)[0]
        return node.metadata["uuid"], node.score

    def save_retrieved(self):
        new = copy.deepcopy(self.RETRIEVED)
        if os.path.exists(PATH["RAG"]["prev_retrieved"]):
            old = load_json_file(PATH["RAG"]["prev_retrieved"])
        else:
            old = {}
        old.update(new)
        self.RETRIEVED = old
        with open(PATH["RAG"]["prev_retrieved"], "w") as f:
            json.dump(self.RETRIEVED, f, indent=4)


if __name__ == "__main__":
    ############################
    #        Single text       #
    ############################
    # text = "In the present study, output from the Scenario Model Intercomparison Project (ScenarioMIP) has been used. ScenarioMIP comprises a primary activity within the context of CMIP6, providing multimodel climate projections based on alternative climate scenarios that are pertinent to societal concerns regarding climate change mitigation, adaptation, or impacts [7]. Te climate projections adopted in Sce-narioMIP are driven by a new set of Shared Socioeconomic Pathways (SSPs) produced with integrated assessment models based on new future pathways of societal development but also incorporating the previously used Representative Concentration Pathways (RCPs) [8]. Te scenario matrix architecture that underlies the framework for formulating new scenarios for climate research is described by van Vuuren et al. [9]; this scenario matrix is based on two main axes: one representing the level of radiative forcing of the climate system (characterized by the RCPs), and the other representing a set of alternative plausible trajectories of future global development (characterized by the SSPs)."
    # print(r.run(text))

    ############################
    #         Parse dir        #
    ############################
    output_dir = PATH["weakly_supervised"]["RAG_preprocessed"]
    doc_dir = PATH["weakly_supervised"]["text"]
    docs = os.listdir(doc_dir)
    nTotal = len(docs)
    # eval_docs = docs[:]
    eval_docs = []
    for doc in docs:
        name = doc.split(".")[0]
        if not os.path.exists(f"{output_dir}/{name}.json") and doc not in eval_docs:
            eval_docs.append(doc)
    print(f"Processing {len(eval_docs)} out of {nTotal} documents")

    i = 0
    r = retriever()
    pbar = tqdm(eval_docs)
    for doc in pbar:
        pbar.set_description(f"{doc} - Total Retrieved: {len(r.RETRIEVED):,}")
        doc_name = doc.split(".")[0]
        i += 1
        with open(f"{doc_dir}/{doc}", "r", encoding="utf-8") as file:
            text = file.read()
        output = r.run(text, tqdm_disable=True, main_pbar=pbar)
        with open(f"{output_dir}/{doc_name}.json", "w") as f:
            json.dump(output, f)
        r.save_retrieved()
    r.save_retrieved()
