# Steam Games Clustering
This project is my first machine learning project as I start my journey into the world of data science!

#### -- Project Status: [Active]

## Project Intro/Objective
The purpose of this project is to find clusters of similar games in terms of reviews and price. I want to know which games are the most bang for my buck.

### Methods Used
* Kmodes clustering
* Linear regression
* Webscraping
* Data Cleaning and feature generation

### Technologies
* Python & Beautifulsoup
* HTML
* IPython & Jupyter Notebook
* Kmodes Package

## Project Description
I am an avid gamer. I always wanted to know if better games cost more money to play, or if they just went on sale less often. Thus, I acquired data for over 25,000 games off the Steam website (https://store.steampowered.com/search/?) to see how many games fall under the "cheaper with bad reviews" category. 

However, due to complications and my inexperience later on into the project, I had to change course and instead ask a different question: To what extent do reviews affect the selling prices of games? This complication, as well as my entire process, will be explained down below.

## Description of my Current Process and Next Steps
Here's a breakdown of the steps I took and the conclusion I arrived at:

I first created a webscraper using Beautifulsoup to scrap over 25,000 games off the steam website. Fields that were scrapped include title, release date, reviews, price, discount, and discounted price.

```python
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
```
* The data was then inputted into a text file and onto a Jupyter Notebook for cleaning. I had some trouble separating the different attributes into columns as the game titles tend to include most separators, such as commas.
* 

## Contact
* feel free to email me at shawnliu30@gmail.com!
