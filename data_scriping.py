import requests
from bs4 import BeautifulSoup
from data_processing import DataProcessor

class ProductWikiScraper:
    def __init__(self, base_url="https://en.wikipedia.org/wiki/"):
        self.base_url = base_url

    def get_wiki_page(self, product_name):
        url = self.base_url + product_name.replace(" ", "_")
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return None

    def parse_wiki_summary(self, page_content):
        soup = BeautifulSoup(page_content, "html.parser")
        paragraphs = soup.find_all("p")
        summary = ""
        for para in paragraphs:
            if para.text.strip():
                summary = para.text.strip()
                break
        return summary if summary else "No summary available."

    def fetch_product_wiki_summary(self, product_name):
        page_content = self.get_wiki_page(product_name)
        if page_content:
            return self.parse_wiki_summary(page_content)
        else:
            return "No Wikipedia page found."
