import requests
from requests_html import HTMLSession
import re
import rich
import rich.table
import urllib.parse
from bs4 import BeautifulSoup
import os
import openai
openai.api_key = "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"
openai.organization = "org-dKb6Se0pRzf3hccYKKg8awJG"

message_history = []
url = 'https://www.ptt.cc/bbs/Gossiping/index.html'
session = HTMLSession()
session.cookies.set('over18', '1')  # 向網站回答滿 18 歲了 !

# 發送 HTTP GET 請求並獲取網頁內容
response = session.get(url)

controls = response.html.find('.action-bar a.btn.wide')

def parse_article_content(url): 
    response = session.get(url)
    main_content = response.html.find('#main-content', first=True).html

    metadata_pattern = r'<div class="article-metaline.*?">.*?</div>'
    main_content_without_metadata = re.sub(metadata_pattern, '', main_content, flags=re.DOTALL)

    push_content_pattern = r'<div class="push"><span class="hl push-tag">. </span><span class="f3 hl push-userid">.*?</span><span class="f3 push-content">:(.*?)</span>.*?</div>'
    push_content_texts = re.findall(push_content_pattern, main_content_without_metadata, re.DOTALL)

    push_content = []
    for push_content_text in push_content_texts:
        push_content.append(push_content_text.strip())

    push_pattern = r'<div class="push">.*?</div>' 
    main_content_without_push = re.sub(push_pattern, '', main_content_without_metadata, flags=re.DOTALL)

    soup = BeautifulSoup(main_content_without_push, 'html.parser')

    # Find all the text within the 'div' tags inside 'main-content' section
    main_text = soup.find('div', id='main-content').get_text()

    # Remove extra line breaks and leading/trailing spaces
    results = '\n'.join(line.strip() for line in main_text.split('\n') if line.strip())

    return results, push_content

def parse_article_entries(elements):
    results = []
    for element in elements:
        push = None
        mark = None
        title = None
        author = None
        date = None
        link = None
        try:
            push = element.find('.nrec', first=True).text
            mark = element.find('.mark', first=True).text
            title = element.find('.title', first=True).text
            author = element.find('.meta > .author', first=True).text
            date = element.find('.meta > .date', first=True).text
            link = element.find('.title > a', first=True).attrs['href']
        except AttributeError:
            # 處理文章被刪除的情況
            if '(本文已被刪除)' in title:
                match_author = re.search('\[(\w*)\]', title)
                if match_author:
                    author = match_author.group(1)
            elif re.search('已被\w*刪除', title):
                match_author = re.search('\<(\w*)\>', title)
                if match_author:
                    author = match_author.group(1)
        # 將解析結果加到回傳的列表中
        results.append({'push': push, 'mark': mark, 'title': title,
                    'author': author, 'date': date, 'link': link})
    return results

def parse_next_link(controls):
    link = controls[1].attrs['href']
    next_page_url = urllib.parse.urljoin('https://www.ptt.cc/', link)
    return next_page_url


def chat(inp, role="user"):
    message_history.append({"role": role, "content": f"{inp}"})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history
    )
    reply_content = completion.choices[0].message.content
    message_history.append({"role": "assistant", "content": f"{reply_content}"})
    return reply_content

def getcontent(artical):    
    content = None
    push_content = None
    content, push_content = parse_article_content("https://www.ptt.cc"+artical['link'])
    artical['content'] = content  # Assuming 'content' is the variable with the content data
    artical['push_content'] = push_content  # Assuming 'push_content' is the list of push content items
    return artical

def getcontent_withURL(URL):    
    content = None
    push_content = None
    content, push_content = parse_article_content(URL)
    return content, push_content

def get_titleNlink():
    # 起始首頁
    url = 'https://www.ptt.cc/bbs/Gossiping/index.html'
    # 想要收集的頁數
    postcnt = 10
    cnt = 0 
    titleNlink = {}

    while cnt < postcnt:
        # 發送 GET 請求並獲取網頁內容
        response = session.get(url)
        # 解析文章列表的元素
        results = parse_article_entries(elements=response.html.find('div.r-ent'))
    #     # 解析下一個連結
        next_page_url = parse_next_link(controls=response.html.find('.action-bar a.btn.wide'))

    #     # 建立表格物件
        for result in results:
            if(result['push'] == "爆"):
                cnt += 1
                result = getcontent(result)

                titleNlink[result['title']] = "https://www.ptt.cc/" + result['link']

        url = next_page_url
    #print(titleNlink)
    return titleNlink


#get_titleNlink()