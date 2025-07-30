#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os


# In[4]:


API_KEY = 'f952a4ba76b442c8ae3f4df662cd0156'  


# In[8]:


# dates for the last 2 days
to_date = datetime.utcnow().date()
from_date = to_date - timedelta(days=2)

QUERY = 'India Pakistan war OR conflict OR border OR attack OR tension'
LANG = 'en'
PAGE_SIZE = 100  # max per page
def fetch_news(query, from_date, to_date, api_key, page_size=100):
    url = 'https://newsapi.org/v2/everything'
    all_articles = []
    for page in range(1, 6):  # try up to 5 pages
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'language': LANG,
            'pageSize': page_size,
            'page': page,
            'sortBy': 'publishedAt',
            'apiKey': API_KEY
        }
        response = requests.get(url, params)
        data = response.json()
        if data.get("status") != "ok":
            print("Error:", data.get("message"))
            break

        articles = data.get("articles", [])
        if not articles:
            break
        all_articles.extend(articles)
    return all_articles


# In[9]:


# fetch articles
articles = fetch_news(QUERY, from_date.isoformat(), to_date.isoformat(), API_KEY)

# convert to dataframe
df = pd.DataFrame([{
    'title': article['title'],
    'description': article['description'],
    'source': article['source']['name'],
    'published_at': article['publishedAt'],
    'url': article['url'],
    'content': article['content']
} for article in articles])

# save to CSV
df.to_csv("india_pakistan_conflict_news.csv", index=False)
print(f"{len(df)} articles saved to 'india_pakistan_conflict_news.csv'")


# In[17]:


import requests
import pandas as pd
from datetime import datetime, timedelta

# API key (hardcoded)
API_KEY = 'f952a4ba76b442c8ae3f4df662cd0156'

# Dates for the last 2 days
to_date = datetime.utcnow().date()
from_date = to_date - timedelta(days=2)

# Query setup
QUERY = 'India Pakistan war OR conflict OR border OR attack OR tension'
LANG = 'en'
PAGE_SIZE = 100  # Max allowed per page

def fetch_news(query, from_date, to_date, api_key, page_size=100):
    url = 'https://newsapi.org/v2/everything'
    all_articles = []

    for page in range(1, 6):  # Try up to 5 pages, but break early if limit is hit
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'language': LANG,
            'pageSize': page_size,
            'page': page,
            'sortBy': 'publishedAt',
            'apiKey': api_key
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data.get("status") != "ok":
            print("Error:", data.get("message"))
            break

        articles = data.get("articles", [])
        if not articles:
            break

        all_articles.extend(articles)

        # Stop if we've hit the free plan limit
        if len(all_articles) >= 100:
            break

    return all_articles[:100]  # Trim to 100 just in case

# Fetch articles
articles = fetch_news(QUERY, from_date.isoformat(), to_date.isoformat(), API_KEY)

# Convert to DataFrame
df = pd.DataFrame([{
    'title': article['title'],
    'description': article['description'],
    'source': article['source']['name'],
    'published_at': article['publishedAt'],
    'url': article['url'],
    'content': article['content']
} for article in articles])

# Save to CSV
df.to_csv("india_pakistan_conflict_news.csv", index=False)
print(f"{len(df)} articles saved to 'india_pakistan_conflict_news.csv'")





# In[18]:


print(df.head())  # Shows the first 5 rows


# In[19]:


import feedparser
import pandas as pd

# Washington Post Technology RSS Feed
rss_url = "http://feeds.washingtonpost.com/rss/business/technology"

# Parse the RSS feed
feed = feedparser.parse(rss_url)

# Filter articles with AI-related keywords
ai_keywords = ["AI", "artificial intelligence", "machine learning", "neural network", "deep learning"]

articles = []
for entry in feed.entries:
    if any(keyword.lower() in entry.title.lower() or keyword.lower() in entry.summary.lower() for keyword in ai_keywords):
        articles.append({
            'title': entry.title,
            'description': entry.summary,
            'published': entry.published,
            'link': entry.link
        })

# Convert to DataFrame
df = pd.DataFrame(articles)

# Save to CSV
df.to_csv("washingtonpost_ai_articles.csv", index=False)
print(f"{len(df)} AI-related articles saved to 'washingtonpost_ai_articles.csv'")


# In[20]:


import feedparser
import pandas as pd

# Multiple RSS feeds to increase coverage
rss_feeds = [
    "http://feeds.washingtonpost.com/rss/business/technology",
    "http://feeds.washingtonpost.com/rss/national",
    "http://feeds.washingtonpost.com/rss/business",
    "http://feeds.washingtonpost.com/rss/business/innovation"
]

# Broader AI-related keywords
ai_keywords = [
    "AI", "artificial intelligence", "machine learning", "neural network", 
    "deep learning", "ChatGPT", "OpenAI", "generative AI", "large language model", "LLM", "chatbot"
]

articles = []
for rss_url in rss_feeds:
    feed = feedparser.parse(rss_url)
    for entry in feed.entries:
        if any(keyword.lower() in (entry.title + entry.get("summary", "")).lower() for keyword in ai_keywords):
            articles.append({
                'title': entry.title,
                'description': entry.get("summary", ""),
                'published': entry.get("published", ""),
                'link': entry.link
            })

# Remove duplicates
df = pd.DataFrame(articles).drop_duplicates(subset="title")

# Save to CSV
df.to_csv("washingtonpost_ai_articles.csv", index=False)
print(f"{len(df)} AI-related articles saved to 'washingtonpost_ai_articles.csv'")



# In[21]:


import requests
from bs4 import BeautifulSoup
import pandas as pd

# Example website that allows scraping: TechCrunch (for educational, low-volume use)
url = "https://techcrunch.com/tag/artificial-intelligence/"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

# Extract articles
articles = []
for post in soup.find_all("article"):
    title_tag = post.find("h2")
    if not title_tag:
        continue

    title = title_tag.get_text(strip=True)
    link_tag = title_tag.find("a")
    link = link_tag["href"] if link_tag else ""
    description_tag = post.find("p")
    description = description_tag.get_text(strip=True) if description_tag else ""

    articles.append({
        "title": title,
        "description": description,
        "link": link
    })

# Convert to DataFrame
df = pd.DataFrame(articles)

# Save to CSV
df.to_csv("techcrunch_ai_articles.csv", index=False)
print(f"{len(df)} AI-related articles saved to 'techcrunch_ai_articles.csv'")


# In[ ]:




