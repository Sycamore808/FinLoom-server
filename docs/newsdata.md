## 个股新闻[](https://akshare.akfamily.xyz/data/stock/stock.html#id139 "Link to this heading")

接口: stock\_news\_em

目标地址: https://so.eastmoney.com/news/s?keyword=603777

描述: 东方财富指定个股的新闻资讯数据

限量: 指定 symbol 当日最近 100 条新闻资讯数据

输入参数

名称

类型

描述

symbol

str

symbol="603777"; 股票代码或其他关键词

输出参数

名称

类型

描述

关键词

object

\-

新闻标题

object

\-

新闻内容

object

\-

发布时间

object

\-

文章来源

object

\-

新闻链接

object

\-

接口示例

import akshare as ak

stock\_news\_em\_df \= ak.stock\_news\_em(symbol\="603777")
print(stock\_news\_em\_df)

数据示例

       关键词  ...                                               新闻链接
0   603777  ...  http://finance.eastmoney.com/a/202506163431529...
1   603777  ...  http://finance.eastmoney.com/a/202506113427724...
2   603777  ...  http://finance.eastmoney.com/a/202506093425572...
3   603777  ...  http://finance.eastmoney.com/a/202506093425584...
4   603777  ...  http://finance.eastmoney.com/a/202506123429086...
..     ...  ...                                                ...
95  603777  ...  http://finance.eastmoney.com/a/202505123401879...
96  603777  ...  http://finance.eastmoney.com/a/202503203351336...
97  603777  ...  http://finance.eastmoney.com/a/202505133403529...
98  603777  ...  http://finance.eastmoney.com/a/202504293392475...
99  603777  ...  http://finance.eastmoney.com/a/202505123402073...
\[100 rows x 6 columns\]

## 财经内容精选[](https://akshare.akfamily.xyz/data/stock/stock.html#id140 "Link to this heading")

接口: stock\_news\_main\_cx

目标地址: https://cxdata.caixin.com/pc/

描述: 财新网-财新数据通-内容精选

限量: 返回所有历史新闻数据

输入参数

名称

类型

描述

\-

\-

\-

输出参数

名称

类型

描述

tag

object

\-

summary

object

\-

interval\_time

object

\-

pub\_time

object

\-

url

object

\-

接口示例

import akshare as ak

stock\_news\_main\_cx\_df \= ak.stock\_news\_main\_cx()
print(stock\_news\_main\_cx\_df)

数据示例

                     tag  ...                                                url
0                 8月5日收盘  ...  https://stock.caixin.com/m/market?cxapp\_link=true
1      高盛上调美国衰退预期但认为风险有限  ...  https://database.caixin.com/2024-08-05/1022233...
2       分析人士：股市逢低买入还为时过早  ...  https://database.caixin.com/2024-08-05/1022233...
3        巴菲特加速减持苹果是重大信号吗  ...  https://www.caixin.com/2024-08-04/102223237.ht...
4                 今日股市热点  ...  https://database.caixin.com/2024-08-05/1022233...
...                  ...  ...                                                ...
12729     （接上条）广电的5G牌怎么打  ...  https://cxdata.caixin.com/twotopic/20200925153...
12730           5G建设最新数据  ...  https://www.caixin.com/2020-11-26/101632865.htm...
12731       银行股尾盘爆发有何道理？  ...  https://database.caixin.com/2020-11-22/10163123...
12732  鸿海或将部分苹果产品生产线迁出中国  ...  https://database.caixin.com/2020-11-27/10163291...
12733          “小酒”股今日普跌  ...  https://database.caixin.com/2020-11-26/10163250...
\[12734 rows x 5 columns\]