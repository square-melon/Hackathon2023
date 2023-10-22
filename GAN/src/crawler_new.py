from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import urllib
import os
import shutil

def crawl(config, args):
    pic_searched_path = config['pic_searched_path']
    pic_searched_path = os.path.join(pic_searched_path, args.gan)
    pic_search_num = config['search_num']
    search_engine = config['search_engine']
    keyword = config['keyword']

    if os.path.exists(pic_searched_path):
        shutil.rmtree(pic_searched_path)

    os.makedirs(pic_searched_path)

    if search_engine == 'google':


        filter = 'filetype:jpg'
        search_keyword = keyword + ' ' + filter
        url = 'https://www.google.com.tw/search?q='+search_keyword+'&tbm=isch'
        options = webdriver.ChromeOptions()
        # options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument("--start-maximized")
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        driver = webdriver.Chrome(options=options)

        img_urls = {}
        driver.get(url)

        for x in range(20):
            driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
            time.sleep(2)
            try:
                button = driver.find_element(By.CLASS_NAME, 'LZ4I')
                button.click()
            except:
                continue

        imgResults = driver.find_elements(By.XPATH,"//img[contains(@class,'Q4LuWd')]")
        cur = -1
        real_num = 0
        while real_num < pic_search_num and cur < len(imgResults):
            cur += 1
            try:
                img_url = imgResults[cur].get_attribute('src')
                if img_url != None and not img_url in img_urls:
                    img_urls[img_url] = ''
                    filename = str(real_num) + '.jpg'
                    print('Download on:', filename)
                    real_num += 1
                    
                    urllib.request.urlretrieve(img_url, os.path.join(pic_searched_path , filename))
            except OSError:
                break
            except:
                continue
        
        # real_num = 0
        # i = 0
        # pos = 0
        # scroll = 1

        # while real_num < pic_search_num:
        #     i += 1
        #     try:
        #         element = driver.find_element(By.XPATH, '//*[@id="islrg"]/div[1]/div[' + str(i) + ']/a[1]/div[1]/img')
        #         img_url = element.get_attribute('src')
        #         if img_url != None and not img_url in img_urls:
        #             img_urls[img_url] = ''
        #             filename = str(real_num) + '.jpg'
        #             print('Download on:', filename)
        #             real_num += 1
        #             time.sleep(1)
                    
        #             urllib.request.urlretrieve(img_url, os.path.join(pic_searched_path , filename))
                        
        #     except OSError:
        #         break
        #     except:
        #         pos += scroll*500
        #         scroll += 1
        #         js = "document.documentElement.scrollTop=%d" % pos
        #         driver.execute_script(js)  
        #         time.sleep(1)
        #         i -= 1
        #         continue
                    
        driver.close()

