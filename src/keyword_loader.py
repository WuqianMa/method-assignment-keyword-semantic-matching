import json
import os


def load_l1_methods(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_l2_methods(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_l1_keywords(l1_methods):
    """Returns {l1_name: [keywords]}"""
    return {m["name"]: m["keywords"] for m in l1_methods}


def get_l2_keywords(l2_methods):
    """Returns {l2_method_name: {"level_1_label": ..., "keywords": [...]}}"""
    return {
        m["method_name"]: {
            "level_1_label": m["level_1_label"],
            "keywords": m["keywords"],
        }
        for m in l2_methods
    }
