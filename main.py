import argparse
import os

import openai

from request import get_abstract, get_papers, make_summary
from utils import get_today, update_paper_list


def main(args):
    # OpenAI API 키 설정
    client = openai.OpenAI(api_key=args.api_key)

    titles_and_abstracts = []
    daily_papers = get_papers()
    for paper_title, paper_url in daily_papers:
        is_updated = update_paper_list(paper_title=paper_title, paper_url=paper_url)
        if is_updated:
            abstract = get_abstract(paper_url=paper_url)
            titles_and_abstracts.append((paper_title, paper_url, abstract))

    # OpenAI API로 요약문 생성
    summaries = [f"## Daily Papers ({get_today()})\n\n"]
    for paper_title, paper_url, abstract in titles_and_abstracts:
        summary = make_summary(
            paper_title=paper_title,
            paper_url=f"https://arxiv.org/abs/{paper_url.split('/papers/')[-1]}",
            abstract=abstract,
            client=client,
        )
        summaries.append(summary + "\n\n")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="")
    args = parser.parse_args()
    main(args)
