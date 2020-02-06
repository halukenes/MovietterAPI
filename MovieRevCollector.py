from selenium import webdriver  
from selenium.webdriver.common.keys import Keys  
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import pandas as pd
import re
import requests
import statistics
import os
import time

folder_path = "./moviereviews/"

for folder in os.listdir(folder_path):

    movies_list = open(folder_path + folder + "/movie_list.txt", "r").read().split('\n')

    for movie in movies_list:
        print(folder + ' - ' + movie)
        review_url = 'https://www.imdb.com/title/' + movie + '/reviews?ref_=tt_urv'

        capa = DesiredCapabilities.CHROME
        capa["pageLoadStrategy"] = "none"
        chrome_options = Options()  
        chrome_options.add_argument("--headless")

        browser = webdriver.Chrome('./chromedriver.exe', chrome_options=chrome_options, desired_capabilities=capa)
        wait = WebDriverWait(browser, 20)
        browser.get(review_url)
        try:
            wait.until(EC.presence_of_element_located((By.ID, 'load-more-trigger')))
            button = browser.find_element_by_id('load-more-trigger')
            try:
                button.click()
            except ElementClickInterceptedException:
                print("ElementClickInterceptedException")
            time.sleep(4)
            button = browser.find_element_by_id('load-more-trigger')
            try:
                button.click()
            except ElementClickInterceptedException:
                print("ElementClickInterceptedException")
            time.sleep(4)
            # wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'lister-list')))
            browser.execute_script("window.stop();")

            soup=BeautifulSoup(browser.page_source)

            contentdiv = soup.find('div', {'class':'lister-list'}).find_all('div', {'class':'content'})

            reviews = []

            for htmlreview in contentdiv:
                reviews.append(htmlreview.find('div').getText())

            df = pd.DataFrame({'Reviews': reviews}) 

            df.to_csv(folder_path + folder + "/" + movie + "_reviews.csv", index=False)
        except TimeoutException:
            print("TimeoutException")

