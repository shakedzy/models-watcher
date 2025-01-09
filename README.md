# 👁️ Models Watcher

[![DOI](https://img.shields.io/badge/Join%20Telegram%20Group!--blue?logo=telegram&style=social)](https://t.me/models_watcher)

This repo contains a script that scans Hugging Face models repository for new and modified base-models (meaning, models which are not derived fom other models),
and send information summary to a dedicated Telegram group.

Unlike the `huggingface_hub`'s `HfApi` object, which cannot tell apart new from modified models, this script analyzes the information
found on Hugging Face's website (such as the model tree and) to extract additional information about each model, thus deciding if it's
new or modified.


## Running yourself
If you'd like to run this script yourself, you'll first need a Bot Token and a Group ID.
Save these as environment variables named `BOT_TOKEN` and `GROUP_CHAT_ID`.

Then install requirements with:
```bash
pip install -r requirements.txt
```

Then, you may run:
```bash
python watcher.py [--days DAYS] [--hours HOURS] [--minutes MINUTES]
```