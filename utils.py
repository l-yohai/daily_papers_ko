from datetime import datetime

import pandas as pd
import pytz

from constant import MARKDOWN_TEMPLATE, MARKDOWN_TEMPLATE_VIDEO, SLACK_TEMPLATE
from request import get_paper_info_per_type


def make_markdown_template(paper_info: dict, summary: str):
    title = get_paper_info_per_type(paper_info, "title")
    paper_url = get_paper_info_per_type(paper_info, "paper_url")
    thumbnail = get_paper_info_per_type(paper_info, "thumbnail")
    upvotes = get_paper_info_per_type(paper_info, "upvotes")
    authors = ", ".join(get_paper_info_per_type(paper_info, "authors"))

    thumbnail_extension = thumbnail.split(".")[-1]

    if thumbnail_extension == "mp4":
        markdown_template = MARKDOWN_TEMPLATE_VIDEO
    else:
        markdown_template = MARKDOWN_TEMPLATE

    return markdown_template.format(
        title=title,
        paper_url=paper_url,
        thumbnail=thumbnail,
        upvotes=upvotes,
        authors=authors,
        summary=summary,
    )


def make_slack_template(paper_info: dict, summary: str):
    title = get_paper_info_per_type(paper_info, "title")
    paper_url = get_paper_info_per_type(paper_info, "paper_url")
    thumbnail = get_paper_info_per_type(paper_info, "thumbnail")
    upvotes = get_paper_info_per_type(paper_info, "upvotes")
    authors = ", ".join(get_paper_info_per_type(paper_info, "authors"))

    return SLACK_TEMPLATE.format(
        title=title,
        paper_url=paper_url,
        thumbnail=thumbnail,
        upvotes=upvotes,
        authors=authors,
        summary=summary,
    )


def get_paper_list(filename: str = "paper_logs/papers_list.csv"):
    df = pd.read_csv(filename)
    return df


def get_today():
    # 오늘 날짜를 YYYY-MM-DD 형식으로 반환
    today = datetime.now(pytz.utc)
    return today.strftime("%Y-%m-%d")
