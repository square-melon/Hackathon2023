from icrawler.builtin import GoogleImageCrawler
import time
import urllib
import os
import shutil

def crawl(config, args):

    # parser
    pic_searched_path = config['pic_searched_path']
    pic_searched_path = os.path.join(pic_searched_path, args.gan)
    pic_gen_path = config['pic_gen_path']
    pic_search_num = config['search_num']
    search_engine = config['search_engine']
    keyword = config['keyword']

    if os.path.exists(pic_searched_path):
        shutil.rmtree(pic_searched_path)

    os.makedirs(pic_searched_path)

    # Google
    if search_engine == 'google':
        google_storage = {'root_dir': pic_searched_path}
        filters = dict(size='=256x256')

        google_crawler = GoogleImageCrawler(
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=4,
            storage=google_storage)
        google_crawler.crawl(keyword=keyword, filters=filters, max_num=pic_search_num, file_idx_offset=0)