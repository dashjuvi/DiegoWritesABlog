import requests
from bs4 import BeautifulSoup
import html
import re

# Define the URL of the page you want to scrape
url = "You_know_how_to_find_it_:)"

# Define headers to mimic a real user request
headers = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "priority": "u=2",
    "referer": "You_know_how_to_find_it_:)",
    "sec-ch-ua": 'whatever',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "script",
    "sec-fetch-mode": "no-cors",
    "sec-fetch-site": "cross-site",
    "user-agent": "WhatEver you want"
}

# Send the GET request
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    print("Request was successful.")
    
    # Parse the content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    anchor_tags = soup.find_all("a", href=True)
    #print(anchor_tags[0].prettify)
    #print("-----------------------")
    urlis= anchor_tags[0]
    get_url=anchor_tags[0].contents[1].contents[1].contents[1].contents[1].contents[1].contents[1].contents[1].contents[1].contents[1].contents[1].contents[0].contents[0]
    #print(get_url.string)

    text_with_url = soup.find(string=re.compile(r"https?://\S+"))
    
    if text_with_url:
        # Extract the URL using regex
        match = re.search(r"https?://[^\s\"<>]+", text_with_url)
        if match:
            url = match.group()
            print("Extracted URL:", url)
        else:
            print("No URL found")
