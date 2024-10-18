MARKDOWN_START = """# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using GPT-4o.

Thanks to [@AK391](https://github.com/AK391) for great work.

"""

MARKDOWN_TEMPLATE = """### [{title}]({paper_url})

![]({thumbnail})

Vote: {upvotes}

Authors: {authors}

{summary}

"""

MARKDOWN_TEMPLATE_VIDEO = """### [{title}]({paper_url})

[Watch Video]({thumbnail})
<div><video controls src=\"{thumbnail}\" muted=\"false\"></video></div>

Vote: {upvotes}

Authors: {authors}

{summary}"""

MARKDOWN_END = """## License

This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.ko) license.

## Citation

If this work is helpful, please kindly cite as:

```bibtex
@misc{daily_papers,
  title = {Huggingface Daily Papers},
  author = {AK391},
  howpublished = {\\url{https://huggingface.co/papers}},
  year = {2023}
}

@misc{daily_papers_ko,
  title = {Automatically translate and summarize huggingface's daily papers into korean},
  author = {l-yohai},
  howpublished = {\\url{https://github.com/l-yohai/daily_papers_ko}},
  year = {2023}
}
```

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=l-yohai/daily_papers_ko&type=Date)
"""

SLACK_START = """{today} Daily Papers (https://huggingface.co/papers?date={today})"""

SLACK_TEMPLATE = """*{title}* ({paper_url})

Vote: {upvotes}

Authors: {authors}

{summary}

Thumbnail: {thumbnail}"""
