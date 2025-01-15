# 👁️ Models Watcher

[![DOI](https://img.shields.io/badge/Join%20Telegram%20Group!--blue?logo=telegram&style=social)](https://t.me/models_watcher)

This repo contains a script that scans Hugging Face models repository for new and modified base-models (models which are not derived fom other models) or models of leading organizations,
and sends information summary to a dedicated Telegram group.

## Which models is it looking for?
To try and narrow down the number of models, the watcher performs several filters:
1. Time based - Only models which were created/modified in a given time-frame
2. `model-index` - Only models with a valid `model-index` 
3. Have dependents _or_ from a leading organization - 
   - Models which are used as base models for other models (fine-tunes, quantizations, etc.)
   - ׁModels from leading organizations

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
