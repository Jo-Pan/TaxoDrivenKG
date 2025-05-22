import json
import re


text_template = "<heading>{}</heading>\n{}\n"


def load_json_file(file_path):
    """
    Loads a JSON file and returns its contents as a Python dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def construct_regex(keyword):
    # split by all special character
    # parts = re.split(r'[-_\s]', keyword)
    parts = re.split(r"[^a-zA-Z0-9]", keyword)
    parts = [p for p in parts if len(p) > 0]

    # allow the special character to be optional
    # out = '_*-*\\s*'.join(parts)
    out = "[^a-zA-Z0-9]*".join(parts)

    # add word boundary to keyword
    return r"\b" + out + r"\b"


def remove_overlapping_mentions(mentions):
    # Sort mentions by start, end in descending order, and length in descending order
    mentions.sort(key=lambda x: x[1] - x[0], reverse=True)

    output = []

    for m in mentions:
        # Check if this mention overlaps with any mention in the output
        if not any(m[0] <= n[1] and m[1] >= n[0] for n in output):
            # If it doesn't, add it to the output
            output.append(m)
    return output


def remove_common_keys(dict1, dict2):
    # This code will remove all items from dict1 if the key appears in dict2.
    return {key: dict1[key] for key in dict1 if key not in dict2}
