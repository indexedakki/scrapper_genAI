import requests
from bs4 import BeautifulSoup

url = "https://indexedakki.github.io/"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all links on the page
links = soup.find_all('a')
for link in links:
    print(link.get('href'))
