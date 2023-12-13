import random
import time

import openai
import requests
from bs4 import BeautifulSoup
from lxml import etree

from utils import get_today

HUGGINGFACE_URL = "https://huggingface.co"
PAPERS_URI = "/papers"
QUERY = "?date={today}"


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
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


@retry_with_exponential_backoff
def make_summary(paper_title, paper_url, abstract, client):
    prompt = f"""Please provide a Korean summary of the abstract for the paper titled '{paper_title}', paper_url is {paper_url}. The abstract is: '{abstract}'. Your summary should distill the key points and main ideas of the abstract, presenting them in a clear and concise manner. Format the summary under the heading '## {paper_title}', followed by a single bullet point. Ensure that the summary is comprehensive yet succinct, capturing the essence of the paper's abstract in a way that is accessible and informative.

The format should be as follows:

## {paper_title}

- summary sentence 1
- summary sentence 2
...

Each sentence in the summary should begin with '-'. The summary should accurately reflect the content and significance of the paper's abstract, providing a clear understanding of its main focus and contributions."""

    response = client.chat.completions.create(
        model="gpt-4-1106-preview", messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content


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
    for el in soup.select(
        "body > div > main > div > section > "
        "div.relative.grid.grid-cols-1.gap-14.lg\:grid-cols-2 > div > article"
    ):
        title = el.find("h3").text.strip()
        url = el.find("a")["href"]

        vote = el.find("div", class_="leading-none").text.strip()

        if not url.startswith("/papers/"):  # video thumbnail
            url = el.find("h3").find("a")["href"]
            thumbnail = el.find("video")["src"]
        else:
            thumbnail = el.select("img")[0]["src"]

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
    tree = etree.HTML(str(soup))
    xpath_exp = "/html/body/div/main/div/section[1]/div/div[2]/p"

    try:
        # 초록을 가져옴
        abstract = tree.xpath(xpath_exp)[0].text

        # 저자를 가져옴
        authors = []
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
    except Exception:
        return None, None

    return abstract, authors


if __name__ == "__main__":
    daily_papers = get_papers()
    print(daily_papers)
