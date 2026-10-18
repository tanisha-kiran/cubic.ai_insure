"""
Import this FIRST in any entry point that should run with no internet.

Hugging Face libraries read these variables when they are imported, and by default they contact the
Hub on startup to check for newer model files (slow or failing with no network, and a privacy leak
for an "offline" product). With them set, models load strictly from the local cache.

Set ALLOW_ONLINE=1 to download models for the first time (e.g. when preparing the offline bundle).
"""
import os

if os.environ.get("ALLOW_ONLINE") != "1":
    for _name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.setdefault(_name, "1")
for _name, _value in (("HF_HUB_DISABLE_TELEMETRY", "1"), ("DO_NOT_TRACK", "1"),
                      ("ANONYMIZED_TELEMETRY", "False"), ("TOKENIZERS_PARALLELISM", "false")):
    os.environ.setdefault(_name, _value)
