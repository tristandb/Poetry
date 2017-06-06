from pprint import pprint

from lxml import html
import requests

"""
    Can scrape PoetryDB.
"""
class Scraper:
    @staticmethod
    def chunk(seq):
        avg = 14
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out

    @staticmethod
    def gethtmlfrompage(url):
        page = requests.get(url)
        tree = html.fromstring(page.content)
        return tree

    @staticmethod
    def getjsonresponse(url):
        api_response = requests.get(url)
        print(api_response.content)
        return api_response.json()


if __name__ == "__main__":
    scraper = Scraper()
    response = Scraper.gethtmlfrompage("http://www.sonnets.org/wyatt.htm")
    titles = response.xpath("//h2/parent::*")
    poems = response.xpath("//dt/parent::*")

    resulterer = []
    pprint(poems)
    for poem in poems:
        if poem.text != '\r\n' and poem.text != '"':
            resulterer.append(poem.text)
    print(len(resulterer))
    print(titles)
    # pprint(resulterer)
    pprint(scraper.chunk(resulterer))
    for a in scraper.chunk(resulterer):
        print(len(a))
