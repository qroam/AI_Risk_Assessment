import os
from bs4 import BeautifulSoup
from lxml import html
import json

# Webpage: https://incidentdatabase.ai/apps/incidents/
# Structure of the html:
"""
1. 事故标题  Incident 1: xxxxxx
//div[@class="titleWrapper"]/div/h1[@data-testid="incident-title"]/text()

//div[strong[contains(text(), "Description")]]/text()

//div[@class="tw-main-container "]

//div[@class="tw-main-container "]/div/div/div/div/div/div[@data-cy="alleged-entities"]

//div[@class="tw-main-container "]/div/div/div
//div[@class="tw-main-container "]
//div[@class="tw-main-container "]
//div[@class="tw-main-container "]
//div[@class="tw-main-container "]

//div[@class="w-fit" and @data-testid="flowbite-tooltip-target"]/p

//div[div[@class="w-fit" and @data-testid="flowbite-tooltip-target"]/p]/div[@data-testid="flowbite-tooltip"]/div/text()

//div[@class="w-fit" and @data-testid="flowbite-tooltip-target"]/p


# id="taxonomy-GMF"
# 表体
//div[@id="taxonomy-GMF"]/div[contains(@class, "grid")]
# 按每一行从左到右排列的每一个表格单元格
//div[@id="taxonomy-GMF"]/div[contains(@class, "grid")]/div
# 按钮
//div[@id="taxonomy-GMF"]/div[contains(@class, "grid")]/button

# id="taxonomy-CSETv0"
# 表体
//div[@id="taxonomy-CSETv0"]/div[contains(@class, "grid")]
# 按每一行从左到右排列的每一个表格单元格
//div[@id="taxonomy-CSETv0"]/div[contains(@class, "grid")]/div
# 按钮
//div[@id="taxonomy-CSETv0"]/div[contains(@class, "grid")]/button

# id="taxonomy-CSETv1"
# 表体
//div[@id="taxonomy-CSETv1"]/div[contains(@class, "grid")]
# 按每一行从左到右排列的每一个表格单元格
//div[@id="taxonomy-CSETv1"]/div[contains(@class, "grid")]/div
# 按钮
//div[@id="taxonomy-CSETv1"]/div[contains(@class, "grid")]/button



//tr[@role="row"]  # 845个

//tr[@role="row"][td[@role="cell"]]
//tr[@role="row"]/td[@role="cell"]  # 5908个. 5908/7 = 844; 除去第一行不是

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
    records = tree.xpath('//tr[@role="row"][td[@role="cell"]]')
    results = []
    for record in records:
        # print(title.text)
        # print(title["href"])
        fields = record.xpath('./td[@role="cell"]')
        assert len(fields) == 7, fields
        
        # for field in fields:
        results.append({
            "Incident ID": fields[0].xpath('string(.)').strip(),
            "title": fields[1].xpath('string(.)').strip(),
            "content": fields[2].xpath('string(.)').strip(),
            "Date": fields[3].xpath('string(.)').strip(),
            "Alleged Deployer of AI System": fields[4].xpath('string(.)').strip(),
            "Alleged Developer of AI System": fields[5].xpath('string(.)').strip(),
            "Alleged Harmed or Nearly Harmed Parties": fields[6].xpath('string(.)').strip(),
        })
    return results


if __name__ == "__main__":
    filename = r"AIID_Incidents_navigator_page.html"
    html_content = load_html(filename)
    records = parse(html_content)
    # print(records)
    
    with open(r"AIID_Incidents_navigator_page.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)
