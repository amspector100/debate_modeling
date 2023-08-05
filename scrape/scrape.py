"""
An extremely hacky way to scrape the data from tabbycat.
But it's not my fault the API is confusing!
"""
import os
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import selenium.common.exceptions
from multiprocessing import Pool
import sys
import time
import tkinter as tk

import utilities

def elapsed(t0):
	return np.around(time.time() - t0, 2)

def copy_data(browser, filename, link=None):
	"""
	Finds the button that copies the table data
	to clipboard and saves it to filename.

	Only works if one has already naved to a page
	with a single one of these buttons.
	"""
	try:
		btn = browser.find_element(
			By.XPATH, 
			"//button[@data-original-title='Copy table data to clipboard in a CSV format']"
		)
	except selenium.common.exceptions.NoSuchElementException as err:
		print(f"For filename={filename}, link={link}, err {err} raised when finding copy CSV button.")
		return None
	btn.click()

	# Save to csv
	root = tk.Tk()
	csv_text = root.clipboard_get()
	with open(filename, "w") as thefile:
		thefile.write(csv_text)
	root.destroy()

def scrape_tournament_heroku(url, tourndir):
	utilities.create_directory(tourndir)

	# Set up webdriver
	options = webdriver.ChromeOptions()
	#options.add_argument('--headless') can't use this, doesn't work
	options.add_argument('--log-level=2')
	browser = webdriver.Chrome(options = options)
	#browser = webdriver.Chrome(ChromeDriverManager().install())
	wait = WebDriverWait(browser, 60)
	browser.get(url)

	# Scrape team tab, speakers, break
	pages = [
		'team tab', 'speaker tab', 'speakers', 'break',
	]
	page_names = {
		"team tab":'tab', "speaker tab":'speakers', 
		"speakers":'speakers', "break":'break'
	}
	for page in pages:
		page_btns = browser.find_elements(By.CLASS_NAME, "nav-link")
		for page_btn in page_btns:
			btn_text = str(page_btn.get_attribute("text")).lower()
			if btn_text == page:
				filename = tourndir + page_names[btn_text] + ".csv"
				page_btn.click()
				copy_data(browser, filename, link=btn_text)
				break

	# Now find round-by-round results tab
	results_drop = browser.find_element(By.ID, "roundsDrop").click()
	# Find list of drop downs
	round_links = []
	round_btns = browser.find_elements(By.CLASS_NAME, "dropdown-item")
	for rbtn in round_btns:
		link = rbtn.get_attribute("href")
		round_links.append(link)

	# loop through and obtain
	for link in round_links:
		# Determine if this is a round, 
		# and if so, which round
		if '_/break' in link:
			continue
		rdname = link.split("/")[-2]
		filename = tourndir + "round" + rdname + ".csv"
		print(f"At filename={filename}, rdname={rdname}, link={link}")

		# navigate and copy data
		browser.get(link)
		copy_data(browser, filename, link=link)

def main():
	t0 = time.time()

	# Read list of tournaments
	tournaments = pd.read_csv("../data/tournaments.csv")
	# Loop through and scrape
	for i in tournaments.index:
		row = tournaments.loc[i]
		code = row['Code']
		tourndir = f"../data/{code}/raw/"
		if os.path.exists(tourndir):
			continue
		else:
			print(f"Scraping {code} at {elapsed(t0)}.")
			scrape_tournament_heroku(row['URL'], tourndir)




if __name__ == "__main__":
	main()