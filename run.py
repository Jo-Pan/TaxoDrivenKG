from consts import *
from utils import *
from utils_RAG import *
from utils_LLM import *
import datetime
import argparse

parser = argparse.ArgumentParser(description="Run climate RAG processing")
parser.add_argument(
    "--experiment", type=str, default="base", help="Name of the experiment"
)  # "0_shot", "no_rag"
parser.add_argument(
    "--model_engine",
    type=str,
    default="Llama-3.3-70B-Instruct",
    help="Name of the model",
)  # "Llama-3.1-8B-Instruct", "DeepSeek-V3", "climategpt-70b "
args = parser.parse_args()

if args.experiment == "base" and args.model_engine == "Llama-3.3-70B-Instruct":
    output_dir = "./outputs"
else:
    output_dir = f"./outputs_exp/{args.experiment}_{args.model_engine}"
os.makedirs(output_dir, exist_ok=True)
print(f"Output dir: {output_dir}")

prompt_dir = f"./outputs_prompts/{args.experiment}"
os.makedirs(prompt_dir, exist_ok=True)
conversations_dir = "./conversations"
today = (
    datetime.date.today().strftime("%Y-%m-%d")
    + f"_{args.experiment}_{args.model_engine}"
)

start_time = datetime.datetime.now()
print(f"Time now: {start_time}")

if __name__ == "__main__":
    MODEL = InfoExtractor(engine=args.model_engine, exp=args.experiment)
    RETRIEVER = retriever(init_model=False)
    TEXTCHUNKER = TextChunker()

    ############################
    #        Single text       #
    ############################
    # file_name = None

    # full_text = "In the present study, output from the Scenario Model Intercomparison Project (ScenarioMIP) has been used. ScenarioMIP comprises a primary activity within the context of CMIP6, providing multimodel climate projections based on alternative climate scenarios that are pertinent to societal concerns regarding climate change mitigation, adaptation, or impacts [7]. Te climate projections adopted in Sce-narioMIP are driven by a new set of Shared Socioeconomic Pathways (SSPs) produced with integrated assessment models based on new future pathways of societal development but also incorporating the previously used Representative Concentration Pathways (RCPs) [8]. Te scenario matrix architecture that underlies the framework for formulating new scenarios for climate research is described by van Vuuren et al. [9]; this scenario matrix is based on two main axes: one representing the level of radiative forcing of the climate system (characterized by the RCPs), and the other representing a set of alternative plausible trajectories of future global development (characterized by the SSPs)."
    # text_chunks = TEXTCHUNKER.get_text_chunks(full_text, file_name=file_name)
    # pbar = tqdm(list(zip(*text_chunks)))

    # outputs = {}
    # conversations = {}
    # chunk_i = 0
    # n_chunks = len(text_chunks[0])
    # for text, start_idx in pbar:
    #     chunk_i += 1
    #     end_idx = start_idx + len(text)
    #     pbar.set_description(
    #         f"File [{file_name}] Chunk [{chunk_i}/{n_chunks}] Retrieving entities"
    #     )
    #     retrieved_nodes = RETRIEVER.run(text, tqdm_disable=True)

    #     pbar.set_description(
    #         f"File [{file_name}] Chunk [{chunk_i}/{n_chunks}] Running MODEL"
    #     )
    #     output, conversation = MODEL.run(text, retrieved_nodes)
    #     outputs[str((start_idx, end_idx))] = output
    #     conversations[str((start_idx, end_idx))] = conversation

    # if file_name is None:
    #     with open(f"{conversations_dir}/{today}.json", "w") as f:
    #         json.dump(conversations, f, indent=4)
    #     with open(f"{output_dir}/{today}.json", "w") as f:
    #         json.dump(outputs, f, indent=4)
    # else:
    #     with open(f"{conversations_dir}/{file_name}.json", "w") as f:
    #         json.dump(outputs, f)
    # exit()

    ############################
    #         Parse dir        #
    ############################
    doc_dir = PATH["weakly_supervised"]["text"]
    docs = os.listdir(doc_dir)
    nTotal = len(docs)
    eval_docs = []
    for doc in docs:
        name = doc.split(".")[0]
        if doc not in eval_docs:
            if not os.path.exists(
                f'{PATH["weakly_supervised"]["RAG_preprocessed"]}/{name}.json'
            ):
                continue
            if not os.path.exists(f"{output_dir}/{name}.json") and not os.path.exists(
                f"./outputs/{name}.json"
            ):
                eval_docs.append(doc)
    print(f"Processing {len(eval_docs)} out of {nTotal} documents")

    pbar = tqdm(eval_docs, leave=False)
    for file in pbar:
        file_name = file.split(".")[0]

        if file.endswith(".txt") or file.endswith(".xmi"):
            with open(f"{doc_dir}/{file_name}.txt", "r") as file:
                full_text = file.read()

            pbar.set_description(f"File [{file_name}] Chunking text")

            text_chunks = TEXTCHUNKER.get_text_chunks(full_text, file_name=file_name)
            pbar2 = tqdm(list(zip(*text_chunks)), disable=True)

            chunk_i = 0
            n_chunks = len(text_chunks[0])
            outputs = {}
            conversations = {}
            prompts = {}

            full_retrieved = None
            if os.path.exists(
                f'{PATH["weakly_supervised"]["RAG_preprocessed"]}/{file_name}.json'
            ):
                full_retrieved = load_json_file(
                    f'{PATH["weakly_supervised"]["RAG_preprocessed"]}/{file_name}.json'
                )
            else:
                exit("No preprocessed file found")

            for text, start_idx in pbar2:
                chunk_i += 1
                end_idx = start_idx + len(text)

                if full_retrieved is not None:
                    pbar.set_description(
                        f"File [{file_name}] Chunk [{chunk_i}/{n_chunks}] Retrieving entities from preprocessed"
                    )
                    # Get noun phrases
                    noun_phrases = RETRIEVER.extract_noun_phrases(text)
                    noun_phrases2 = RETRIEVER.filter_noun_phrases(noun_phrases)
                    retrieved_nodes = {}
                    for phrase in noun_phrases2:
                        if phrase in full_retrieved:
                            if full_retrieved[phrase] != []:
                                retrieved_nodes[phrase] = full_retrieved[phrase]
                # else:
                #     pbar.set_description(
                #         f"File [{file_name}] Chunk [{chunk_i}/{n_chunks}] Retrieving entities"
                #     )
                #     retrieved_nodes = RETRIEVER.run(text, tqdm_disable=True)

                pbar.set_description(
                    f"File [{file_name}] Chunk [{chunk_i}/{n_chunks}] Running LLM"
                )
                output, conversation = MODEL.run(text, retrieved_nodes)
                outputs[str((start_idx, end_idx))] = output
                conversations[str((start_idx, end_idx))] = conversation
                prompts[str((start_idx, end_idx))] = conversation[0]["content"]

            with open(f"{output_dir}/{file_name}.json", "w") as f:
                json.dump(outputs, f, indent=4)

            if not os.path.exists(f"{prompt_dir}/{file_name}.json"):
                with open(f"{prompt_dir}/{file_name}.json", "w") as f:
                    json.dump(prompts, f, indent=4)

            if today:
                with open(f"{conversations_dir}/{today}_{file_name}.json", "w") as f:
                    json.dump(conversations, f, indent=4)
                today = None
    print(
        f"Time now: {datetime.datetime.now()}. Time elapsed: {datetime.datetime.now() - start_time}"
    )
