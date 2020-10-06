import requests as rs
import bs4
import time
import csv

url_root = "https://movie.naver.com"


def get_year_url(year):
    return "https://movie.naver.com/movie/sdb/browsing/bmovie.nhn?open=" + year


def get_year_code(year, URL):
    """
    연도에 개봉한 영화의 영화 코드 수집. url 만들 때 사용
    :param year: 연도
    :param URL: base url
    :return: 영화 코드, 영화 url 딕셔너리
    """
    movies = dict()
    print("Crawling year " + year)

    #   for page in range(1,44) 44 page is max
    for page in range(1, 44):
        if page % 20 == 0:
            print(page)
        response = rs.get(URL + "&page=" + str(page))
        html_content = response.text.encode(response.encoding)
        if html_content == None:
            break
        navigator = bs4.BeautifulSoup(
            html_content, 'html.parser', from_encoding='utf-8')
        navigator = navigator.find("ul", {"class": "directory_list"})
        navigator = navigator.find_all("li")
        for n in navigator:
            t = n.find("a")
            title = t.text
            movie_url = url_root + t.get("href")
            movies[title] = movie_url
        time.sleep(0.1)

    return movies


def crawl_movie_list_year(year):
    """
    연도의 개봉한 영화 리스트 크롤링
    :param year: 연도
    """
    URL = get_year_url(year)
    out_two = get_year_code(year, URL)
    with open(year + "_file.tsv", 'w', encoding='utf-8', newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["movie", "url", "code"])
        for title, movie_url in out_two.items():
            code = movie_url.split("=")[-1]
            if "code=" in movie_url:
                writer.writerow([title, movie_url, code])


def get_demo_info(code):
    """
    영화 코드에 해당하는 영화의 예매율 정보 추출
    :param code: 영화 코드
    :return: 연령별, 성별 예매율
    """
    print("get demo info of " + code)
    base_url = "https://movie.naver.com/movie/bi/mi/basic.nhn?code=" + code
    response = rs.get(base_url)

    html_content = response.text.encode(response.encoding)
    navigator = bs4.BeautifulSoup(
        html_content, 'html.parser', from_encoding='utf-8')
    title = navigator.find("div", {"class": "graph_wrap"})
    if title == None:
        print("found no graph info")
        return [], []

    response_text = response.text
    idx = response_text.find("sPer")
    rate_one = response_text[idx + 7:idx + 9]
    rate_one_b = response_text[idx + 7:idx + 10]
    if rate_one == "0 ":
        male_rate = 0.0
    elif rate_one_b == "100":
        male_rate = 100.0 * 0.01
    else:
        male_rate = float(response_text[idx + 7:idx + 9]) * 0.01

    idx_two = response_text.find("sPer", idx + 10, len(response_text))
    rate_one = response_text[idx_two + 7:idx_two + 9]
    rate_one_b = response_text[idx_two + 7:idx_two + 10]
    if rate_one == "0 ":
        female_rate = 0.0
    elif rate_one_b == "100":
        female_rate = 100.0 * 0.01
    else:
        female_rate = float(response_text[idx_two + 7:idx_two + 9]) * 0.01

    bar_graph = title.find("div", {"class": "bar_graph"})
    ages = bar_graph.find_all("strong", {"class": "graph_percent"})

    gender = [male_rate, female_rate]
    ages = []
    for age in ages:
        ages.append(float(age.text[:-1]) * 0.01)

    time.sleep(0.1)

    return gender, ages


def download_demo_info_year(year):
    """
    연도의 개봉 영화 예매율 크롤링
    :param year: 연도
    """
    with open(year + "_file.tsv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        rows = []
        total_genders = []
        total_ages = []
        for i, row in enumerate(reader):
            # if i > 10:
            #    break
            rows.append(row)
            code = row[2]
            genders, ages = get_demo_info(code)
            total_genders.append(genders)
            total_ages.append(ages)

    print("Writing Demo info to file....")
    with open(year + "_file_demo_two.tsv", 'w', encoding='utf-8') as f:
        f.write("movie\turl\tcode\tmale\tfemale\t10\t20\t30\t40\t50\n")
        for i, row in enumerate(rows):
            if len(total_genders[i]) != 0:
                f.write(row[0] + "\t" + row[1] + "\t" + row[2] + "\t" + str(total_genders[i][0]) + "\t" + str(
                    total_genders[i][1])
                        + "\t" + str(total_ages[i][0]) + "\t" + str(total_ages[i][1]) + "\t" + str(total_ages[i][2])
                        + "\t" + str(total_ages[i][3]) + "\t" + str(total_ages[i][4]) + "\n")


if __name__ == "__main__":
    year = "2019"
    # 연도에 개봉한 영화 이름과 코드 다운로드
    crawl_movie_list_year(year)
    # 연도에 개봉한 영화 예매율 정보 크롤링
    download_demo_info_year(year)