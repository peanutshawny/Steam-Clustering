from urllib.request import urlopen
from bs4 import BeautifulSoup as soup
import datetime
from time import sleep
import re

# makes soup object out of web-page


def make_soup(url):
    page = urlopen(url)
    soup_data = soup(page, 'html.parser')
    return soup_data


# makes the soup out of url input
steam_soup = make_soup('https://store.steampowered.com/search/?ignore_preferences=1')

today = datetime.datetime.today()

# making the file
file_name = f'steam_sales {today:%B, %d, %Y}.txt'
f = open(file_name, 'w', encoding='utf-8')

headers = 'title, date, reviews, discount, original price, discounted price\n'

f.write(headers)

# for loop to run through any number of pages, index must start at 2
for num in range(2, 2429):
    next_page = steam_soup.find('a', attrs={'class': 'pagebtn'}).get('href')

    # replaces last element of next_page (the page number) with next page so it doesn't go back to previous page
    # regex designed to find the last number of the url, which is the page number
    next_page = next_page.replace(re.search('\d+$', next_page).group(0), str(num))

    # insert container code
    containers = steam_soup.findAll('div', {'class': 'responsive_search_name_combined'})

    # for loop to find and write all elements of every container into the csv
    for container in containers:
        title = container.div.span.text
        date = container.find('div', {'class': 'col search_released responsive_secondrow'}).text

        try:
            reviews = (container.find('div', {'class': 'col search_reviewscore responsive_secondrow'}).span[
                           'data-tooltip-html'].split('<br>'))[0]
        except Exception as e:
            reviews = ''

        discount = container.find('div', {'class': 'col search_price_discount_combined responsive_secondrow'}). \
            find('div', {'class': 'col search_discount responsive_secondrow'}).text.strip()

        try:
            original_price = container.find('div', {'class': 'col search_price_discount_combined responsive_secondrow'}). \
                find('div', {'class': 'col search_price responsive_secondrow'}).text.strip()
        except Exception as e:
            try:
                original_price = container.find('div',
                                                {'class': 'col search_price_discount_combined responsive_secondrow'}). \
                    find('div', {'class': 'col search_price discounted responsive_secondrow'}).span.strike.text
            except Exception as e:
                original_price = ''
        try:
            discounted_price = container.find('div',
                                              {'class': 'col search_price_discount_combined responsive_secondrow'}). \
                find('div', {'class': 'col search_price discounted responsive_secondrow'}).text.split('C')
            discounted_price = 'C' + discounted_price[2]
        except Exception as e:
            discounted_price = ''

        print('title', title)
        print('date', date)
        print('reviews', reviews)
        print('discount', discount)
        print('original price', original_price)
        print('discounted price', discounted_price)
        print('')

        f.write(title + '*' + date.replace(',', '') + '*' + reviews + '*' + discount + '*' + original_price + '*' +
                discounted_price + '\n')

    print(num)

    # sleep(2)
    steam_soup = make_soup(next_page)

f.close()



