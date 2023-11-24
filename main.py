import argparse
import os

import openai

from request import get_abstract_and_authors, get_papers, make_summary
from slackbot import make_template_for_slackbot, send
from utils import get_today, update_paper_list


def main(args):
    # OpenAI API 키 설정
    client = openai.OpenAI(api_key=args.api_key)

    target_papers = []

    daily_papers = get_papers()

    for daily_paper in daily_papers:
        paper_title = daily_paper["title"]
        thumbnail = daily_paper["thumbnail"]
        paper_url = daily_paper["url"]
        is_updated = update_paper_list(paper_title=paper_title, paper_url=paper_url)
        if is_updated:
            abstract, authors = get_abstract_and_authors(paper_url=paper_url)
            target_papers.append((paper_title, thumbnail, authors, paper_url, abstract))

    # 업데이트된 논문이 있을 경우
    if len(target_papers) > 0:
        # OpenAI API로 요약문 생성
        summaries = [f"## Daily Papers ({get_today()})\n\n"]
        to_slack_summaries = []
        for paper_title, thumbnail, authors, paper_url, abstract in target_papers:
            summary = make_summary(
                paper_title=paper_title,
                paper_url=f"https://arxiv.org/abs/{paper_url.split('/papers/')[-1]}",
                abstract=abstract,
                client=client,
            )
            summary = "\n".join(
                [line for line in summary.split("\n") if line.startswith("-")]
            )
            if thumbnail.split(".")[-1] == "mp4":
                summary = f"### [{paper_title}](https://arxiv.org/abs/{paper_url.split('/papers/')[-1]})\n\n[Watch Video]{thumbnail}\n<div><video controls src=\"{thumbnail}\" muted=\"false\"></video></div>\n\nAuthors: {', '.join(authors)}\n\n{summary}"
            else:
                summary = f"### [{paper_title}](https://arxiv.org/abs/{paper_url.split('/papers/')[-1]})\n\n![]({thumbnail})\n\nAuthors: {', '.join(authors)}\n\n{summary}"
            summaries.append(summary + "\n\n")
            to_slack_summary = make_template_for_slackbot(summary)
            to_slack_summaries.append(to_slack_summary)

        yy, mm, dd = get_today().split("-")
        os.makedirs(f"paper_logs/{yy}/{mm}", exist_ok=True)

        with open(f"paper_logs/{yy}/{mm}/{dd}.md", "w") as f:
            f.writelines(summaries)

        with open("readme_start", "r") as f:
            start_content = f.read()
        with open("readme_end", "r") as f:
            end_content = f.read()
        with open("README.md", "w") as f:
            f.writelines(start_content + "\n\n")
            f.writelines(summaries)
            f.writelines("\n\n")
            f.writelines(end_content)

        send(args, to_slack_summaries)


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

    main(args)
