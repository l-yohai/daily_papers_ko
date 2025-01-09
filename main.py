import argparse
import asyncio
import json
import os

import openai
from tqdm import tqdm

from constant import MARKDOWN_END, MARKDOWN_START
from request import get_paper_info_per_type, get_papers, make_summary_async
from slackbot import send
from utils import get_paper_list, get_today, make_markdown_template, make_slack_template


async def process_paper_async(client, paper_info, paper_list):
    title = get_paper_info_per_type(paper_info, "title")
    paper_url = get_paper_info_per_type(paper_info, "paper_url")

    if title in paper_list["title"].values:
        print(f"{title} is already in the list")
        return None, None, None

    abstract = get_paper_info_per_type(paper_info, "abstract")

    response_summary = await make_summary_async(
        client=client,
        title=title,
        paper_url=paper_url,
        abstract=abstract,
    )

    try:
        summary = json.loads(response_summary)
    except Exception as e:
        print(f"Error: {e}\n{response_summary}")
        return None, None, None

    summary = "\n".join([f"- **{k}**: {v}" for k, v in summary.items()])

    markdown_summary = make_markdown_template(paper_info=paper_info, summary=summary)

    slack_summary = make_slack_template(paper_info=paper_info, summary=summary)

    return paper_info, markdown_summary, slack_summary


async def main(args):
    client = openai.AsyncOpenAI(api_key=args.api_key)

    today = get_today()
    markdown_summaries = [f"## Daily Papers ({get_today()})\n\n"]
    slack_summaries = []

    daily_papers = get_papers()

    paper_list = get_paper_list()

    new_papers = []

    tasks = [
        process_paper_async(client, paper_info, paper_list)
        for paper_info in daily_papers
    ]

    for task in tqdm(asyncio.as_completed(tasks), total=len(daily_papers)):
        paper_info, markdown_summary, slack_summary = await task
        if paper_info:
            new_papers.append((paper_info, markdown_summary, slack_summary))

    if new_papers:
        # Sort new_papers by upvotes
        sorted_new_papers = sorted(new_papers, key=lambda x: x[0]["paper"]["upvotes"], reverse=True)

        # Separate sorted papers into their respective summaries
        for new_paper_info, markdown_summary, slack_summary in sorted_new_papers:
            paper_list.loc[len(paper_list)] = [
                get_today(),
                get_paper_info_per_type(new_paper_info, "title"),
                get_paper_info_per_type(new_paper_info, "paper_url"),
            ]
            markdown_summaries.append(markdown_summary)
            slack_summaries.append(slack_summary)

        paper_list.to_csv("paper_logs/papers_list.csv", index=False, encoding="utf-8")

        yy, mm, dd = get_today().split("-")
        os.makedirs(f"paper_logs/{yy}/{mm}", exist_ok=True)

        with open(f"paper_logs/{yy}/{mm}/{dd}.md", "a") as f:
            f.writelines(markdown_summaries)

        with open("README.md", "w") as f:
            f.writelines(MARKDOWN_START.format(today=today))
            f.writelines(markdown_summaries)
            f.writelines(MARKDOWN_END)

        send(args, slack_summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="")
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

    asyncio.run(main(args))
