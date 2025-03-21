
import requests
from bs4 import BeautifulSoup
import csv
import time
import re
from urllib.parse import urljoin

def scrape_diegowritesa_blog():
    url = "[ChooseYourURLHERE]"
    headers = {"User-Agent": "Mozilla/5.0"} # Or any agent you want, watch out to not get blocked by the website or detected as malicious activity
    
    try:
        response = requests.get(url, headers=headers)  # Use requests or anything you want to do the web request
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")  # Create the soup
        
        # Extract the first link with [CTI] in the text
        cti_links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True) if "[CTI]" in a.get_text()]  # Find my tag, change this to however your link looks like
        
        if cti_links:
            print(f"Extracted CTI Links: {cti_links[0]}")  # Debugging (only print the first link)
        return cti_links[1] if cti_links else None
    
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def extract_hashes(text):
    # Extract hashes: MD5 (32 chars), SHA-1 (40 chars), SHA-256 (64 chars), SHA-512 (128 chars)
    hashes = re.findall(r'\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b|\b[a-fA-F0-9]{128}\b', text)
    return hashes

def extract_ips(text):
    # Regex pattern for IPv4 and IPv6
    ips = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b', text)
    return ips

def extract_domains(text):
    # Regex pattern for domains
    domains = re.findall(r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b', text)
    return domains

def scrape_cti_data(url):
    data = []
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        time.sleep(1)  # Prevent overwhelming the server
        print(f"Requesting: {url}")  # Debugging
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        content_type = response.headers.get("Content-Type", "")
        if "text/plain" in content_type or "csv" in content_type:
            body_text = response.text  # Read raw text content
        else:
            soup = BeautifulSoup(response.text, "html.parser")
            body_text = "\n".join([element.get_text() for element in soup.find_all()])
        
        # Extract hashes, IPs, and domains
        hashes = extract_hashes(body_text)
        ips = extract_ips(body_text)
        domains = extract_domains(body_text)
        
        # Only add the link if hashes, IPs, or domains are found
        if hashes or ips or domains:
            full_content = f"URL: {url}\n\nContent:\n{body_text}\n\nExtracted Hashes:\n{', '.join(hashes)}\nExtracted IPs:\n{', '.join(ips)}\nExtracted Domains:\n{', '.join(domains)}"
            data.append([url, hashes, ips, domains, full_content])
            print(f"Hashes found for {url}: {', '.join(hashes)}")
            print(f"IPs found for {url}: {', '.join(ips)}")
            print(f"Domains found for {url}: {', '.join(domains)}")
        else:
            print(f"No hashes, IPs, or domains found for {url}, skipping...")
    
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
    
    return data

def save_to_csv(data, filename="cti_links_url.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["URL", "Hashes", "IPs", "Domains", "Full Content"])
        writer.writerows(data)

if __name__ == "__main__":
    first_cti_url = scrape_diegowritesa_blog()
    if first_cti_url:
        scraped_data = scrape_cti_data(first_cti_url)
        save_to_csv(scraped_data)
        print(f"Data saved to cti_links_data_first_url.csv")
