import os
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
            recent_models.append(model.modelId)
        else:
            break  
    logger.info(f"Found {len(recent_models)} total models from relevant time-threshold on hugging-face")
    return recent_models


def analyze_model_tree(model_id: str, time_threshold: datetime) -> Model | None:
    response = requests.get(f"https://huggingface.co/{model_id}/tree/main")
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    for ul in soup.find_all("ul"):
        if ul.find_all("time"):
            files: list[ModelFile] = []
            for li in ul.find_all("li"):
                filename = li.find("a").get_text(strip=True)
                is_directory = any(['text-blue' in c for c in li.find("svg").get("class")])
                change_time = datetime.strptime(li.find("time").get("datetime"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                files.append(ModelFile(filename=filename, is_directory=is_directory, change_time=change_time))
            is_new = all([f.change_time >= time_threshold for f in files])
            return Model(model_id=model_id, files=files, is_new=is_new)  
    return None


def filter_base_models(models: list[str], time_threshold: datetime) -> list[Model]:
    base_models: list[Model] = []
    for model_id in tqdm(models, desc="Analyzing models"):
        is_base = None
        try:
            response = requests.get(f"https://huggingface.co/{model_id}")
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            for h2 in soup.find_all("h2"):
                if not "Model tree for" in h2.get_text(strip=True): continue   
                div = h2.find_next_sibling("div")
                if div:
                    div_text = div.get_text(strip=True)
                    is_base = "Base model" not in div_text
                if is_base:
                    model = analyze_model_tree(model_id, time_threshold)
                    if model:
                        base_models.append(model)
                break
        except Exception as e:
            logger.error(f"Failed to analyze model {model_id} - {e.__class__}: {e}")     
    logger.info(f"Found {len(base_models)} base models from models list")
    return base_models     


def escape_markdown(text: str, 
                    *, 
                    chars: list[str] = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
                    ) -> str:
    for char in chars:
        text = text.replace(char, f"\\{char}")
    return text


def prepare_message(*, days: int = 0, hours: int = 0, minutes: int = 0) -> str:
    assert any([days, hours, minutes]), "At least one of the time arguments must be greater than 0"
    delta = timedelta(days=days, hours=hours, minutes=minutes)
    time_threshold = datetime.now(timezone.utc) - delta
    model_ids = find_latest_models(time_threshold)
    models = filter_base_models(model_ids, time_threshold)
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
    if message:
        lb = [("days", days), ("hours", hours), ("minutes", minutes)]
        lb = [t for t in lb if t[1] > 0]
        lb = [(t[0][:-1], t[0]) if t[0] == 1 else (t[0], t[0]) for t in lb]
        if len(lb) == 1:
            look_back = f"{lb[0][1]} {lb[0][0]}"
        elif len(lb) == 2:
            look_back = f"{lb[0][1]} {lb[0][0]} and {lb[1][1]} {lb[1][0]}"
        else:
            look_back = f"{lb[0][1]} {lb[0][0]}, {lb[1][1]} {lb[1][0]} and {lb[2][1]} {lb[2][0]}"
        message += f"\n🔍 _Looking back {look_back}\\._"
    return message.strip()


async def send_group_message(message: str):
    bot = Bot(token=BOT_TOKEN)
    await bot.send_message(chat_id=GROUP_CHAT_ID, text=message, parse_mode="MarkdownV2")


def main():                    
    parser = ArgumentParser()
    parser.add_argument("--days", type=int, default=0, help="Number of days to look back")
    parser.add_argument("--hours", type=int, default=0, help="Number of hours to look back")
    parser.add_argument("--minutes", type=int, default=0, help="Number of minutes to look back")
    args = parser.parse_args()

    message = prepare_message(days=args.days, hours=args.hours, minutes=args.minutes)
    if message:
        logger.info(f"Sending message:\n{message}")
        asyncio.run(send_group_message(message))
    else:
        logger.info("No message to send")


if __name__ == "__main__":
    main()
