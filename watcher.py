import os
import logging
import asyncio
import requests
from telegram import Bot
from fnmatch import fnmatch
from bs4 import BeautifulSoup
from dataclasses import dataclass
from argparse import ArgumentParser
from collections import defaultdict
from huggingface_hub import HfApi, ModelInfo
from datetime import datetime, timedelta, timezone


BOT_TOKEN = os.environ['BOT_TOKEN']
GROUP_CHAT_ID = os.environ['GROUP_CHAT_ID']


logger = logging.getLogger("models-watcher")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
hf_api = HfApi()


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


def get_top_organizations(max_orgs: int = 100) -> list[str]:
    models = hf_api.list_models(sort='downloads', direction=-1, limit=max_orgs * 5, full=True)
    author_stats = defaultdict(lambda: {'model_count': 0, 'total_downloads': 0})
    for model in models:
        author = model.author
        downloads = model.downloads if hasattr(model, 'downloads') and model.downloads else 0
        author_stats[author]['model_count'] += 1
        author_stats[author]['total_downloads'] += downloads
    sorted_authors = sorted(author_stats.items(), key=lambda x: x[1]['total_downloads'], reverse=True)
    top_authors: list[str] = [author for author, _ in sorted_authors]
    
    top_orgs: list[str] = []
    for author in top_authors:
        try: 
            hf_api.get_user_overview(author, token=False)
            continue
        except:
            top_orgs.append(author)
        finally:
            if len(top_orgs) >= max_orgs:
                break

    logger.info(f"Retrieved {len(top_orgs)} top organizations from hugging-face:\n{top_orgs}")
    return top_orgs


def find_model_files(model_id: str) -> list[ModelFile]:
    FILES_TO_IGNORE = ['.git*', '*.md', 'config.json']

    response = requests.get(f"https://huggingface.co/{model_id}/tree/main")
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    modified_files: list[ModelFile] = []

    # This section looks into the model repository page. 
    # The <ul> tag examined here is the table seen on the page, 
    # and each <li> is a list in the table
    for ul in soup.find_all("ul"):

        # Only one <ul> has <time> tags in it, which are the last modified times of the files. 
        # This is the table we're looking for
        if ul.find_all("time"):
            
            for li in ul.find_all("li"):
                try:
                    # First <a> tag in <li> is the filename
                    filename: str = li.find("a").get_text(strip=True)  
                    if any(fnmatch(filename, pattern) for pattern in FILES_TO_IGNORE): continue
                    # the first <svg> is the icon next to the filename. 
                    # If it has the class "text-blue", it's a directory icon.
                    is_directory = any(['text-blue' in c for c in li.find("svg").get("class")]) 
                    # The <time> tag is the last modified time of the file
                    change_time = datetime.strptime(li.find("time").get("datetime"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                    modified_files.append(ModelFile(filename=filename, is_directory=is_directory, change_time=change_time))
                
                except Exception as e:
                    logger.error(f'Error analyzing file from repo webpage - {e.__class__.__name__}: {e}\n{li}')

    return modified_files


def keep_single_model(model: ModelInfo) -> bool:
    card_data = model.card_data
    if card_data:
        base_model = card_data.base_model
        return not base_model
    else:
        return False


def find_new_models(time_threshold: datetime, top_orgs: list[str]) -> list[Model]:
    models = hf_api.list_models(sort="created_at", direction=-1, full=True, filter="model-index")
    recent_models: list[str] = []
    for model in models:
        if model.created_at is None:
            continue
        elif model.created_at >= time_threshold:
            if model.author in top_orgs or keep_single_model(model):
                recent_models.append(model.id)
        else:
            break  
    logger.info(f"Found {len(recent_models)} new models on hugging-face")
    return [Model(m, files=[], is_new=True) for m in recent_models]


def find_modified_models(time_threshold: datetime, top_orgs: list[str]) -> list[Model]:
    models = hf_api.list_models(sort="last_modified", direction=-1, full=True, filter="model-index")
    recent_models: list[Model] = []  
    for model in models:
        if model.last_modified is None or model.created_at is None:
            continue
        elif model.last_modified >= time_threshold:
            if model.created_at < time_threshold:
                if model.author in top_orgs or keep_single_model(model):
                    files = find_model_files(model.id)
                    if any(f.change_time >= time_threshold for f in files):
                        recent_models.append(
                            Model(
                                model_id=model.id, 
                                files=files,
                                is_new=all(f.change_time >= time_threshold for f in files)
                            )
                        )
        else:
            break  
    logger.info(f"Found {len(recent_models)} modified models on hugging-face")
    return recent_models


def escape_markdown(text: str, 
                    *, 
                    chars: list[str] = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
                    ) -> str:
    for char in chars:
        text = text.replace(char, f"\\{char}")
    return text


def prepare_message(time_threshold: datetime) -> str:
    top_orgs = get_top_organizations()
    new_models = find_new_models(time_threshold, top_orgs)
    all_modified_models = find_modified_models(time_threshold, top_orgs)

    modified_models: list[Model] = []
    for model in all_modified_models:
        if model.is_new:
            new_models.append(model)
        else:
            modified_models.append(model)

    message = ""
    if new_models:
        message += f"🆕 *New models:*\n"
        for model in new_models:
            message += f" • [{escape_markdown(model.model_id)}](https://huggingface.co/{escape_markdown(model.model_id, chars=['('])})\n"
    if modified_models:
        message += f"\n🔄 *Modified models:*\n"
        for model in modified_models:
            modified_files = [f"{'Contents of ' if f.is_directory else ''}`{f.filename}{'/' if f.is_directory else ''}`" 
                              for f in model.files if f.change_time >= time_threshold]
            modified_files = [escape_markdown(f) for f in modified_files]
            message += f" • [{escape_markdown(model.model_id)}](https://huggingface.co/{escape_markdown(model.model_id, chars=['('])}) _\\(Updated files: {', '.join(modified_files)}\\)_\n"
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

    assert any([args.days, args.hours, args.minutes]), "At least one of the time arguments (days/hours/minutes) must be greater than 0"
    delta = timedelta(days=args.days, hours=args.hours, minutes=args.minutes)
    time_threshold = datetime.now(timezone.utc) - delta

    message = prepare_message(time_threshold)
    if message:
        logger.info(f"Sending message:\n{message}")
        asyncio.run(send_group_message(message))
    else:
        logger.info("No message to send")


if __name__ == "__main__":
    main()
