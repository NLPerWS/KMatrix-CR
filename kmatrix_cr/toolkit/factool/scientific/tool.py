import yaml

from scholarly import scholarly
from scholarly import ProxyGenerator
import os

class google_scholar():
    def __init__(self):
        self.pg = ProxyGenerator()
        self.scraper_api_key = os.environ.get("SCRAPER_API_KEY", None)
        # assert scraper_api_key is not None, "Please set the SCRAPER_API_KEY environment variable."
        # assert scraper_api_key != '', "Please set the SCRAPER_API_KEY environment variable."
        # success = pg.ScraperAPI(self.scraper_api_key)
        # scholarly.use_proxy(pg)

    def run(self, query):
        assert self.scraper_api_key is not None, "Please set the SCRAPER_API_KEY environment variable."
        assert self.scraper_api_key != '', "Please set the SCRAPER_API_KEY environment variable."
        scholarly.use_proxy(self.pg)
        
        try:
            results = scholarly.search_pubs(query)
            paper_info = next(results)
            paper_info_subset = {key: paper_info['bib'][key] for key in ['title', 'author', 'pub_year']}
            return paper_info_subset
        except StopIteration:
            return {'title': "no match!", "author": "no match!", "pub_year": "no match!"}
