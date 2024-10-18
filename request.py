from datetime import datetime

import aiohttp
import fitz
import openai
import pytz
import requests
import tiktoken

from prompts import SYSTEM_PROMPT

PAPER_PAGES_API = "https://huggingface.co/api/daily_papers"


async def get_paper_full_content_async(paper_url: str, session: aiohttp.ClientSession):
    async with session.get(paper_url.replace("abs", "pdf")) as response:
        content = await response.read()

    with open("tmp.pdf", "wb") as f:
        f.write(content)

    encoding = tiktoken.encoding_for_model("gpt-4o")
    doc = fitz.open("tmp.pdf")
    full_content = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        full_content += text

    encoded_text = encoding.encode(full_content)
    if len(encoded_text) > 100_000:
        encoded_text = encoded_text[:100_000]
        full_content = encoding.decode(encoded_text)

    return full_content


async def make_summary_async(client: openai.AsyncOpenAI, **kwargs):
    async with aiohttp.ClientSession() as session:
        full_content = await get_paper_full_content_async(kwargs["paper_url"], session)

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    title=kwargs["title"],
                    paper=full_content,
                    abstract=kwargs["abstract"],
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    return response.choices[0].message.content.strip()


def get_papers() -> list[dict]:
    """오늘의 논문 목록을 반환합니다.

    Returns:
        list[dict]: 논문 목록
        {
            "title": 논문 제목,
            "url": 논문 페이지 URL,
            "thumbnail": 논문 썸네일 URL,
            "vote": 논문 추천 수
        }
    """

    papers = requests.get(PAPER_PAGES_API).json()
    return papers


def get_paper_info_per_type(paper_info: dict, info_type: str) -> str | list[str] | int:
    """논문 정보를 가져오는 통합 함수입니다.

    Args:
        paper_info (dict): 논문의 정보
        info_type (str): 가져올 정보 유형 ('published_at', 'abstract', 'authors', 'title', 'upvotes', 'thumbnail', 'paper_url')

    Returns:
        str | list[str] | int: 요청된 정보
    """
    if info_type == "published_at":
        published_at = paper_info.get("publishedAt", "")
        published_at = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%S.%fZ")
        published_at = published_at.astimezone(pytz.timezone("Asia/Seoul"))
        return published_at.strftime("%Y-%m-%d")
    elif info_type == "abstract":
        return paper_info.get("paper", {}).get("abstract", "")
    elif info_type == "authors":
        authors = paper_info.get("paper", {}).get("authors", [])
        return list(set(author.get("name", "") for author in authors))
    elif info_type == "title":
        return " ".join(paper_info.get("paper", {}).get("title", "").split())
    elif info_type == "upvotes":
        return paper_info.get("paper", {}).get("upvotes", 0)
    elif info_type == "thumbnail":
        return paper_info.get("thumbnail", "")
    elif info_type == "paper_url":
        paper_id = paper_info.get("paper", {}).get("id", "")
        return f"https://arxiv.org/abs/{paper_id}"
    else:
        raise ValueError(f"Unknown info_type: {info_type}")


if __name__ == "__main__":
    papers = get_papers()
    print(f"length: {len(papers)}")
    print(papers[0])
