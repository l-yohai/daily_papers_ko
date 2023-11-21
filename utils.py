from datetime import datetime

import pandas as pd
import pytz


def get_today():
    # 오늘 날짜를 YYYY-MM-DD 형식으로 반환
    today = datetime.now(pytz.timezone("Asia/Seoul"))
    return today.strftime("%Y-%m-%d")


def update_paper_list(paper_title, paper_url, filename="paper_logs/papers_list.csv"):
    exists_paper = False

    df = pd.read_csv(filename)

    # 논문 제목이 DataFrame에 없는 경우 추가
    if paper_title not in df["title"].values:
        titles = df["title"].values.tolist()
        urls = df["url"].values.tolist()
        titles.append(paper_title)
        urls.append(f"https://arxiv.org/abs/{paper_url.split('/papers/')[-1]}")
        df = pd.DataFrame(data={"title": titles, "url": urls})
        df.to_csv(filename, index=False, encoding="utf-8")
        exists_paper = False
        print(f"Added '{paper_title}' to {filename}")
    else:
        exists_paper = True
        print(f"'{paper_title}' already exists in {filename}")

    return not exists_paper
