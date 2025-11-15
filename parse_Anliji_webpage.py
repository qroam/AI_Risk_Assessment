import os
from bs4 import BeautifulSoup
from lxml import html

# Webpage: https://www.ai-governance.online/cases-cn/
# Structure of the html:
"""
//div[@class="card-body"]
//div[@class="card-body"]/h5/text()  # 标题
//div[@class="card-body"]/p[@class="card-text"]/text()  # 内容。注，似乎有些条目没有内容
//div[@class="card-body"]/a/span[@class="badge badge-case-tag"]/text()  # 分类标签
"""

def load_html(filename):
    """
    Load HTML content from a file.
    """
    if not os.path.exists(filename):
        return None
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()
        
def parse(html_content):
    tree = html.fromstring(html_content)
    records = tree.xpath('//div[@class="card-body"]')
    results = []
    for record in records:
        # print(title.text)
        # print(title["href"])
        title = record.xpath('./h5/text()')
        assert len(title) == 1, title
        title = title[0].strip()

        content = record.xpath('./p[@class="card-text"]/text()')
        if len(content) == 0:
            print(f"record has no content: {title}")
            content = None
        else:
            assert len(content) == 1
            content = content[0].strip()
        
        tags = record.xpath('./a/span[@class="badge badge-case-tag"]/text()')
        results.append({
            "title": title,
            "content": content,
            "tags": tags,
        })
    return results


if __name__ == "__main__":
    filename = r"人工智能风险与治理案例库 _ 人工智能治理公共服务平台.html"
    html_content = load_html(filename)
    records = parse(html_content)
    
