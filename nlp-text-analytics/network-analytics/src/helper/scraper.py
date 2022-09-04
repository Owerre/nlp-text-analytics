###########################
# Author: S. A. Owerre
# Date modified: 09/06/2021
# Class: Web scraping
###########################

# filter warnings
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import ads


class Scraper:
    """A class to scrape and preprocess arXiv papers."""

    def __init__(self):
        pass

    def get_data(self, arxiv='physics:cond-mat'):
        """This function scrape the condensed matter physics papers,
        and create a pandas dataframe of the data from BeautifulSoup xml parser.
        """

        base_url = 'http://export.arxiv.org/oai2?verb=ListRecords&'
        url = (
            base_url
            + 'from=2017-01-01&until=2019-12-31&'
            + 'metadataPrefix=arXiv&set=%s' % arxiv
        )

        df = pd.DataFrame()  # Empty Pandas DataFrame

        while True:

            print('fetching', url)

            try:
                req_xml = urlopen(url)

            except HTTPError as e:
                if e.code == 503:
                    to = int(e.hdrs.get('retry-after', 30))
                    print('Got 503. Retrying after {0:d} seconds.'.format(to))
                    time.sleep(to)
                    continue

                else:
                    raise

            b_soup = BeautifulSoup(req_xml, 'xml')
            token = b_soup.resumptionToken
            for record in b_soup.find_all('record'):
                titles = record.find('title').text
                author = [
                    a.text.strip('\n') for a in record.find_all('author')
                ]
                abstracts = record.find('abstract').text
                categories = record.find('categories').text
                arxiv_ids = record.find('id').text
                created = record.find('created').text
                doi = record.find('doi')
                comments = record.find('comments')

                if doi != None:
                    doi = doi.text
                if comments != None:
                    comments = comments.text

                content = {
                    'abstract': abstracts,
                    'authors': author,
                    'title': titles,
                    'categories': categories.split(),
                    'arXiv_id': arxiv_ids,
                    'date_created': created,
                    'doi': doi,
                    'comments': comments,
                }

                df = df.append(content, ignore_index=True)

            if token is None or token.text is None:
                break
            else:
                url = base_url + 'resumptionToken=%s' % (token.text)
        return df