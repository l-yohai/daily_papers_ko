import random
import requests
import time

from bs4 import BeautifulSoup
from lxml import etree
import openai

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
            except errors as e:
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

### [paper title](paper_url)

- summary

Each sentence in the summary should begin with '-'. The summary should accurately reflect the content and significance of the paper's abstract, providing a clear understanding of its main focus and contributions."""

    response = client.chat.completions.create(
        model="gpt-4-1106-preview", messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content


def get_papers():
    # 오늘의 논문 목록을 반환
    today = get_today()
    url = HUGGINGFACE_URL + PAPERS_URI + QUERY.format(today=today)

    # 논문 목록 페이지를 가져옴
    response = requests.get(url)

    # 논문 목록 페이지를 파싱
    soup = BeautifulSoup(response.text, "html.parser")

    daily_papers = []
    for el in soup.find_all("h3"):
        daily_papers.append((el.get_text(strip=True), el.find("a")["href"]))

    return daily_papers


def get_abstract(paper_url):
    # 논문 페이지에서 초록을 가져옴
    url = HUGGINGFACE_URL + paper_url

    # 논문 페이지를 가져옴
    response = requests.get(url)

    # 논문 페이지를 파싱
    soup = BeautifulSoup(response.text, "html.parser")
    tree = etree.HTML(str(soup))
    xpath_exp = "/html/body/div/main/div/section[1]/div/div[2]/p"

    # 초록을 가져옴
    abstract = tree.xpath(xpath_exp)[0].text

    return abstract
