import random
import time

import openai
import requests
from bs4 import BeautifulSoup
import tiktoken

from utils import get_today
from prompts import SYSTEM_PROMPT_SUMMARIZATION

HUGGINGFACE_URL = "https://huggingface.co"
PAPERS_URI = "/papers"
QUERY = "?date={today}"


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 2,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


def get_paper_full_content(paper_url):
    for trial in range(3):
        try:
            paper_page = requests.get(paper_url)
            if paper_page.status_code == 200:
                break
        except requests.exceptions.ConnectionError as e:
            print(e)
            time.sleep(trial * 30 + 15)
    paper_soup = BeautifulSoup(paper_page.text, "html.parser")

    sections = paper_soup.find_all("section")
    section_dict = {}

    for section in sections:
        section_id = section.get("id")
        if section_id:
            # <h2> 태그 내에서 제목 찾기
            title_tag = section.find("h2")
            if title_tag:
                # <span> 태그 내용 제거
                if title_tag.find("span"):
                    title_tag.span.decompose()
                section_title = title_tag.text.strip()
            else:
                section_title = "No title found"

            # 섹션의 전체 텍스트 내용을 추출 (제목 제외)
            section_content = "\n".join(
                [para.text.strip() for para in section.find_all("p")]
            )

            # 사전에 섹션 ID, 제목, 내용 저장
            section_dict[section_id] = {
                "title": section_title,
                "content": section_content,
            }

    return section_dict


def truncate_text(text):
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return encoding.decode(
        encoding.encode(
            text,
            allowed_special={"<|endoftext|>"},
        )[:2048]
    )


@retry_with_exponential_backoff
def make_summary(full_content, client):
    summarization_input = ""
    if type(full_content) is not str:
        for _, section in full_content.items():
            if section["title"] == "No title found":
                continue
            if section["content"] == "":
                continue

            summarization_input += (
                f"Section: {section['title']}\n{section['content']}\n\n"
            )

    summarization_input = truncate_text(summarization_input)

    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARIZATION},
            {"role": "user", "content": f"""{summarization_input}"""},
        ],
        response_format={
            "type": "json_object",
        },
    )
    return response.choices[0].message.content.strip()


def get_url():
    today = get_today()
    url = HUGGINGFACE_URL + PAPERS_URI + QUERY.format(today=today)
    return url


def get_papers():
    # 오늘의 논문 목록을 반환
    today = get_today()
    url = HUGGINGFACE_URL + PAPERS_URI + QUERY.format(today=today)

    # 논문 목록 페이지를 가져옴
    response = requests.get(url)

    # 논문 목록 페이지를 파싱
    soup = BeautifulSoup(response.text, "html.parser")

    daily_papers = []

    papers = soup.find_all("article", class_="relative flex flex-col overflow-hidden rounded-xl border")
    for paper in papers:
        title = paper.find("h3").text.strip()
        url = paper.find("a")["href"]
        vote = paper.find_all("div", class_="leading-none")[-1].text.strip()
        if not url.startswith("/papers/"):  # video thumbnail
            url = paper.find("h3").find("a")["href"]
            thumbnail = paper.find("video")["src"]
        else:
            thumbnail = paper.select("img")[0]["src"]

        daily_papers.append(
            {
                "title": title,
                "url": url,
                "thumbnail": thumbnail,
                "vote": vote,
            }
        )

    return daily_papers


def get_abstract_and_authors(paper_url):
    # 논문 페이지에서 초록을 가져옴
    url = HUGGINGFACE_URL + paper_url

    # 논문 페이지를 가져옴
    response = requests.get(url)

    # 논문 페이지를 파싱
    soup = BeautifulSoup(response.text, "html.parser")

    # 초록을 가져옴
    abstract = soup.select(
        "body > div > main > div > section.pt-8.border-gray-100.md\:col-span-7.sm\:pb-16.lg\:pb-24.relative > div > div.pb-8.pr-4.md\:pr-16 > p"
    )[0].text

    # 저자를 가져옴
    authors = []
    for el in soup.select(
        "body > div > main > div > "
        "section.pt-8.border-gray-100.md\:col-span-7.sm\:pb-16.lg\:pb-24.relative > "
        "div > div.pb-10.md\:pt-3 > "
        "div.relative.flex.flex-wrap.items-center.gap-2.text-base.leading-tight"
    )[0].find_all("span"):
        author = ""
        if el:
            if el.find("a") is not None:
                author = el.find("a").text.strip()
            if not author and el is not None:
                if el.text.strip() != "" and el.text.strip() != "Authors:":
                    author = el.text.split("\n")[0].split("\t")[0].strip()

        if author:
            authors.append(author)

    authors = list(set(authors))
    return abstract, authors


if __name__ == "__main__":
    daily_papers = get_papers()
    print(daily_papers)
