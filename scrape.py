import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Create a folder to store books
os.makedirs("books", exist_ok=True)

# Base URL for Project Gutenberg
BASE_URL = "https://www.gutenberg.org"

# Search URL with filters for English books
SEARCH_URL = f"{BASE_URL}/ebooks/search/?lang=en"

# Function to download a book as a text file
def download_book(book_url, book_title):
    try:
        response = requests.get(book_url, timeout=10)
        response.raise_for_status()
        file_path = os.path.join("books", f"{book_title}.txt")
        
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(response.text)
        print(f"Downloaded: {book_title}")
    except Exception as e:
        print(f"Failed to download {book_title}: {e}")

# Function to scrape books from the search page
def scrape_books_from_search():
    page = 1
    while True:
        try:
            print(f"Scraping page {page}...")
            response = requests.get(f"{SEARCH_URL}&page={page}", timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            # Find all book links on the page
            book_links = soup.select("li.booklink a.link")
            if not book_links:
                print("No more books found.")
                break
            
            for link in tqdm(book_links, desc=f"Downloading books from page {page}"):
                book_title = link.text.strip()
                if '(Chinese)' in book_title: continue
                book_page_url = BASE_URL + link["href"]

                # Get the book's main page to find the plain text file
                book_page_response = requests.get(book_page_url, timeout=10)
                book_page_response.raise_for_status()
                book_soup = BeautifulSoup(book_page_response.content, "html.parser")

                # Find the plain text file link
                txt_link = book_soup.find("a", href=True, text="Plain Text UTF-8")

                if txt_link:
                    book_txt_url = BASE_URL + txt_link["href"]
                    download_book(book_txt_url, book_title)
                else:
                    print(f"No plain text file found for {book_title}.")

            page += 1
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break

if __name__ == "__main__":
    scrape_books_from_search()
