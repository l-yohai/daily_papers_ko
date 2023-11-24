import argparse
import re

from slack_sdk import WebClient

from utils import get_today

yy, mm, dd = get_today().split("-")


def get_daily_papers():
    with open(f"paper_logs/{yy}/{mm}/{dd}.md", "r") as f:
        daily_papers = f.read()
    return daily_papers


def make_template_for_slackbot(daily_papers: str):
    templates = []

    contents = daily_papers.split("###")[1:]

    for content in contents:
        # 0: 제목, 1: 썸네일, 2: 저자, 3: 요약
        title_and_url, thumbnail, authors, summary = content.split("\n\n")[:4]
        title_and_url = title_and_url.strip().replace("[", "*").replace("]", "* ")

        if "video" in thumbnail:
            pattern = r'<video[^>]*src="([^"]+)"[^>]*>'
            match = re.search(pattern, thumbnail)
            if match:
                thumbnail = match.group(1)
            else:
                thumbnail = ""
        elif "png" in thumbnail:
            pattern = r"!\[\]\(([^)]+)\)"
            match = re.search(pattern, thumbnail)
            if match:
                thumbnail = match.group(1)
            else:
                thumbnail = ""

        template = f"""{title_and_url}\n\nAuthors: {authors}\n\n{summary}\n\nThumbnail: {thumbnail}\n\n"""

        templates.append(template)

    return templates


def send(args, templates=[]):
    sc = WebClient(token=args.slack_api_token)

    if not templates:
        daily_papers = get_daily_papers()
        templates = make_template_for_slackbot(daily_papers=daily_papers)

    sc.chat_postMessage(
        channel=args.target_channel_name,
        text=f"{yy}-{mm}-{dd} Daily Papers",
    )
    for template in templates:
        # get the timestamp of the parent messagew
        result = sc.conversations_history(channel=args.target_channel_id)
        conversation_history = result["messages"]  # [0] is the most recent message
        message_ts = conversation_history[0]["ts"]

        sc.chat_postMessage(
            channel=args.target_channel_name,
            text=template,
            thread_ts=message_ts,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace_name", type=str, default="", help="Slack workspace name"
    )
    parser.add_argument(
        "--target_channel_name", type=str, default="", help="Target channel name"
    )
    parser.add_argument(
        "--target_channel_id", type=str, default="", help="Target channel ID"
    )
    parser.add_argument(
        "--slack_api_token", type=str, default="", help="Slack API token"
    )
    args = parser.parse_args()

    send(args)
