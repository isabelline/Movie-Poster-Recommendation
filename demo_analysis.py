import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import operator
import pandas as pd


def count_gender_avg(df):
    """
    성별 평균 예매율 계산
    :param df: 예매율 정보가 있는 Dataframe
    :return: 성별 예매율 누적합
    """
    male_cnt = 0
    female_cnt = 0
    for i, row in df.iterrows():
        male = row['male']
        female = row['female']
        male_cnt += male
        female_cnt += female
    return [male_cnt, female_cnt]


def count_age_avg(df):
    """
    연령별 예매율 누적합
    :param df: 예매율 정보가 있는 Dataframe
    :return: 연령별 예매율 누적합
    """
    ten_cnt = 0
    ten_cnt_two = 0
    ten_cnt_three = 0
    ten_cnt_four = 0
    ten_cnt_five = 0
    for i, row in df.iterrows():
        ten = row['10']
        ten_two = row['20']
        ten_three = row['30']
        ten_four = row['40']
        ten_five = row['50']
        ten_cnt += ten
        ten_cnt_two += ten_two
        ten_cnt_three += ten_three
        ten_cnt_four += ten_four
        ten_cnt_five += ten_five
    return [ten_cnt, ten_cnt_two, ten_cnt_three, ten_cnt_four, ten_cnt_five]


def count_gender_proportion(df):
    """
    남성/ 여성의 예매율이 더 높은 영화 비율 추출
    :param df: 예매율 정보가 있는 Dataframe
    :return: 남성/ 여성의 예매율이 더 높은 영화 비율
    """
    male_cnt = 0
    female_cnt = 0
    for i, row in df.iterrows():
        male = row['male']
        female = row['female']
        if male > female:
            male_cnt += 1
        else:
            female_cnt += 1
    return [x / float(len(df)) for x in [male_cnt, female_cnt]]


def count_age_proportion(df):
    """
    연령대별 예매율이 높은 영화 비율 추출
    :param df: 예매율 정보가 있는 Dataframe
    :return: 연령대별 예매율이 높은 영화 비율
    """
    ten_cnt = 0
    ten_cnt_two = 0
    ten_cnt_three = 0
    ten_cnt_four = 0
    ten_cnt_five = 0
    for i, row in df.iterrows():
        ten = row['10']
        ten_two = row['20']
        ten_three = row['30']
        ten_four = row['40']
        ten_five = row['50']
        items = [ten, ten_two, ten_three, ten_four, ten_five]
        index, value = max(enumerate(items), key=operator.itemgetter(1))
        if index == 0:
            ten_cnt += 1
        elif index == 1:
            ten_cnt_two += 1
        elif index == 2:
            ten_cnt_three += 1
        elif index == 3:
            ten_cnt_three += 1
        else:
            ten_cnt_four += 1
    return [x / float(len(df)) for x in [ten_cnt, ten_cnt_two, ten_cnt_three, ten_cnt_four, ten_cnt_five]]


def plot_donut_chart(labels, data, title):
    """
    도넛 차트 그리기
    :param labels: 라벨 리스트
    :param data: 값 리스트
    :param title: 그래프 제목
    """
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    recipe = labels
    data = data

    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(recipe[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)

    ax.set_title(title)

    plt.show()


def func(pct, allvals):
    absolute = int(pct / 100. * np.sum(allvals))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)


def plot_circle_chart(labels, data, title, category):
    """
    원 그래프 그리기
    :param labels: 라벨 리스트
    :param data: 값 리스트
    :param title: 그래프 제목
    :param category: 그래프 레전드 카테고리 이름
    """
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    data = data
    ingredients = labels

    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: '{:.2f}'.format(pct),
                                      textprops=dict(color="w"))

    path = 'C:/Windows/Fonts/H2GTRM.TTF'
    fontprop = fm.FontProperties(fname=path, size=10)

    font_name = fontprop.get_name()
    # plt.legend(prop={'family': font_name, 'size': 20})

    legend = ax.legend(wedges, ingredients,
                       title=title,
                       loc="center left",
                       bbox_to_anchor=(1, 0, 0.5, 1), prop={'family': font_name, 'size': 8})

    legend.set_title(category, prop=fontprop)
    fontprop_two = fm.FontProperties(size=6)

    plt.setp(autotexts, size=8, weight="bold", fontproperties=fontprop_two)

    ax.set_title(title, fontproperties=fontprop)

    plt.show()


# 크롤링한 성별, 연령대 예매율 파일
df = pd.read_csv("C://Users/SKbroadband/Documents/2019_file_demo_two.tsv", sep="\t", encoding='utf-8')

# age avg
data = count_age_avg(df)
plot_circle_chart(['10대', '20대', '30대', '40대', '50대'], data, "나이별 예매율 평균", "나이대")

# age proportion
data = count_age_proportion(df)
plot_circle_chart(['10대', '20대', '30대', '40대', '50대'], data, "가장 예매율이 높은 나이대 비율", "나이대")

# gender avg
data = count_gender_avg(df)
plot_circle_chart(['남자', '여자'], data, "성별 예매율 평균", "성별")

# gender proportion
data = count_gender_proportion(df)
plot_circle_chart(['남자', '여자'], data, "가장 예매율이 높은 성별 비율", "성별")