import os
import json
import logging
import asyncio
import requests
from tqdm import tqdm
from telegram import Bot
from bs4 import BeautifulSoup
from huggingface_hub import HfApi
from dataclasses import dataclass
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone


BOT_TOKEN = os.environ['BOT_TOKEN']
GROUP_CHAT_ID = os.environ['GROUP_CHAT_ID']


logger = logging.getLogger("models-watcher")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


@dataclass
class ModelFile:
    filename: str
    is_directory: bool
    change_time: datetime


@dataclass
class Model:
    model_id: str
    files: list[ModelFile]
    is_new: bool


def find_latest_models(time_threshold: datetime) -> list[str]:
    api = HfApi()
    models = api.list_models(sort="lastModified", direction=-1, full=True, filter="model-index")
    recent_models = []
    for model in models:
        if model.lastModified is None:
            continue
        last_modified = model.lastModified
        if last_modified >= time_threshold:
            recent_models.append(model.modelId)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            break  
    logger.info(f"Found {len(recent_models)} total models from relevant time-threshold on hugging-face")
    return recent_models


def analyze_model_tree(model_id: str, time_threshold: datetime) -> Model | None:
    response = requests.get(f"https://huggingface.co/{model_id}/tree/main")
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    # This section looks into the model repository page. 
    # The <ul> tag examined here is the table seen on the page, 
    # and each <li> is a list in the table
    for ul in soup.find_all("ul"):

        # Only one <ul> has <time> tags in it, which are the last modified times of the files. 
        # This is the table we're looking for
        if ul.find_all("time"):
            files: list[ModelFile] = []
            for li in ul.find_all("li"):
                # First <a> tag in <li> is the filename
                filename = li.find("a").get_text(strip=True)  
                # the first <svg> is the icon next to the filename. 
                # If it has the class "text-blue", it's a directory icon.
                is_directory = any(['text-blue' in c for c in li.find("svg").get("class")]) 
                # The <time> tag is the last modified time of the file
                change_time = datetime.strptime(li.find("time").get("datetime"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                files.append(ModelFile(filename=filename, is_directory=is_directory, change_time=change_time))

            # A model is considered new if *all* files are newer than the time threshold
            is_new = all([f.change_time >= time_threshold for f in files])
            return Model(model_id=model_id, files=files, is_new=is_new)  
    return None


def filter_base_models(models: list[str], 
                       undecided_models: list[str],
                       time_threshold: datetime,
                       grace_time_threshold: datetime
                       ) -> tuple[list[Model], list[str]]:
    base_models: list[Model] = []
    new_undecided_models: list[str] = []
    for model_id in tqdm(models + undecided_models, desc="Analyzing models"):
        is_base = None
        try:
            response = requests.get(f"https://huggingface.co/{model_id}")
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            for h2 in soup.find_all("h2"):
                # Looking at the model tree shown on the model's page
                # This section shows the relations between the current model and other known models
                # If exists, it's always found on the <div> below a <h2> starting with the text "Model tree for"
                if not "Model tree for" in h2.get_text(strip=True): continue   
                div = h2.find_next_sibling("div")
                if div:
                    div_text = div.get_text(strip=True)
                    # On Hugging Face model page, if the model tree displays a model
                    # with the text "Base model", it's always *another* model, meaning 
                    # the current model is a derived model and not a base model.
                    # Base model always has a tree, but without the text "Base model" in it 
                    # (it has the model's name in it).
                    is_base = "Base model" not in div_text
                if is_base:
                    model = analyze_model_tree(model_id, time_threshold=grace_time_threshold if model_id in undecided_models else time_threshold) 
                    if model:
                        base_models.append(model)
                break

        except Exception as e:
            logger.error(f"Failed to analyze model {model_id} - {e.__class__}: {e}") 

        finally:
            if is_base is None:
                # In the case where the <div> of the model tree is not found,
                # the model is placed in the undecided models list, and will be checked again later.
                # This is to avoid false negatives, as the model might be a base model but the tree is not loaded yet.
                new_undecided_models.append(model_id)

    logger.info(f"Found {len(base_models)} base models from models list, {len(new_undecided_models)} undecided models")
    return base_models, new_undecided_models     


def escape_markdown(text: str, 
                    *, 
                    chars: list[str] = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
                    ) -> str:
    for char in chars:
        text = text.replace(char, f"\\{char}")
    return text


def prepare_message(*, undecided_model_ids: list[str], time_threshold: datetime, grace_time_threshold: datetime) -> tuple[str, list[str]]:
    model_ids = find_latest_models(time_threshold)
    models, new_undecided_models = filter_base_models(model_ids,undecided_model_ids, time_threshold, grace_time_threshold)
    message = ""
    new_models = [m for m in models if m.is_new]
    modified_models = [m for m in models if not m.is_new]
    if new_models:
        message += f"🆕 *New models:*\n"
        for model in new_models:
            message += f" • [{escape_markdown(model.model_id)}](https://huggingface.co/{escape_markdown(model.model_id, chars=['('])})\n"
    if modified_models:
        message += f"\n🔄 *Modified models:*\n"
        for model in modified_models:
            modified_files = [f"{'Content of ' if f.is_directory else ''}{f.filename}{'/' if f.is_directory else ''}" for f in model.files if f.change_time >= time_threshold]
            modified_files = [escape_markdown(f) for f in modified_files]
            message += f" • [{escape_markdown(model.model_id)}](https://huggingface.co/{escape_markdown(model.model_id, chars=['('])}) _\\(Updated files: {', '.join(modified_files)}\\)_\n"
    return message.strip(), new_undecided_models


def load_undecided_models(grace_time_threshold: datetime) -> dict[str, datetime]:
    loaded_models = {}
    try:
        with open("undecided_models.json", "r") as f:
            models = {k: datetime.strptime(v, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc) for k, v in json.load(f).items()}
            models = {k: v for k,v in models.items() if v >= grace_time_threshold}
            loaded_models = models
    except Exception as e:
        logger.error(f"Failed to load undecided models - {e.__class__}: {e}")
    finally:
        logger.info(f"Loaded {len(loaded_models)} undecided models:\n{loaded_models}")
        return loaded_models
    

def save_undecided_models(existing_undecided_models: dict[str, datetime], new_undecided_models: dict[str, datetime]) -> None:
    undecided_models = new_undecided_models | existing_undecided_models
    undecided_models = {k: datetime.strftime(v, "%Y-%m-%dT%H:%M:%S") for k,v in undecided_models.items()}
    try:
        with open("undecided_models.json", "w") as f:
            json.dump(undecided_models, f)
        logger.info(f"Saved {len(undecided_models)} undecided models:\n{undecided_models}")
    except Exception as e:
        logger.error(f"Failed to save {len(undecided_models)} undecided models - {e.__class__}: {e}")


async def send_group_message(message: str):
    bot = Bot(token=BOT_TOKEN)
    await bot.send_message(chat_id=GROUP_CHAT_ID, text=message, parse_mode="MarkdownV2")


def main():                    
    parser = ArgumentParser()
    parser.add_argument("--days", type=int, default=0, help="Number of days to look back")
    parser.add_argument("--hours", type=int, default=0, help="Number of hours to look back")
    parser.add_argument("--minutes", type=int, default=0, help="Number of minutes to look back")
    parser.add_argument("--grace-time", dest='grace', type=int, default=5, help="Number of hours of grace time for undecided models")
    args = parser.parse_args()

    assert any([args.days, args.hours, args.minutes]), "At least one of the time arguments (days/hours/minutes) must be greater than 0"
    delta = timedelta(days=args.days, hours=args.hours, minutes=args.minutes)
    time_threshold = datetime.now(timezone.utc) - delta
    grace_delta = timedelta(hours=args.grace)
    grace_time_threshold = time_threshold - grace_delta

    undecided_models = load_undecided_models(grace_time_threshold)
    logger.info(f"Loaded {len(undecided_models)} undecided models")
    message, new_undecided_models_list = prepare_message(undecided_model_ids=list(undecided_models.keys()), time_threshold=time_threshold, grace_time_threshold=grace_time_threshold)
    save_undecided_models(undecided_models, {k: datetime.now(timezone.utc) for k in new_undecided_models_list})

    if message:
        logger.info(f"Sending message:\n{message}")
        asyncio.run(send_group_message(message))
    else:
        logger.info("No message to send")


if __name__ == "__main__":
    main()
