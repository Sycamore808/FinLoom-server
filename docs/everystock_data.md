#### 上海证券交易所-每日概况[](https://akshare.akfamily.xyz/data/stock/stock.html#id8 "Link to this heading")

接口: stock\_sse\_deal\_daily

目标地址: http://www.sse.com.cn/market/stockdata/overview/day/

描述: 上海证券交易所-数据-股票数据-成交概况-股票成交概况-每日股票情况

限量: 单次返回指定日期的每日概况数据, 当前交易日数据需要在收盘后获取; 注意仅支持获取在 20211227（包含）之后的数据

输入参数

名称

类型

描述

date

str

date="20250221"; 当前交易日的数据需要交易所收盘后统计; 注意仅支持获取在 20211227（包含）之后的数据

输出参数

名称

类型

描述

单日情况

object

包含了网页所有字段

股票

float64

\-

主板A

float64

\-

主板B

float64

\-

科创板

float64

\-

股票回购

float64

\-

接口示例

import akshare as ak

stock\_sse\_deal\_daily\_df \= ak.stock\_sse\_deal\_daily(date\="20250221")
print(stock\_sse\_deal\_daily\_df)

数据示例

    单日情况           股票          主板A       主板B         科创板  股票回购
0    挂牌数    2321.0000    1693.0000   43.0000    585.0000   0.0
1   市价总值  529981.4800  456997.7000  942.6300  72041.1500   0.0
2   流通市值  501613.5100  445348.4700  713.7700  55551.2700   0.0
3   成交金额    8561.3100    6413.6300    4.3000   2143.3700   0.3
4    成交量     608.5800     556.5800    0.7200     51.2900   0.1
5  平均市盈率      14.3200      13.2000    7.1600     45.7800   NaN
6    换手率       1.6154       1.4034    0.4565      2.9752   0.0
7  流通换手率       1.7068       1.4401    0.6029      3.8584   0.0

### 个股信息查询-东财[](https://akshare.akfamily.xyz/data/stock/stock.html#id9 "Link to this heading")

接口: stock\_individual\_info\_em

目标地址: http://quote.eastmoney.com/concept/sh603777.html?from=classic

描述: 东方财富-个股-股票信息

限量: 单次返回指定 symbol 的个股信息

输入参数

名称

类型

描述

symbol

str

symbol="603777"; 股票代码

timeout

float

timeout=None; 默认不设置超时参数

输出参数

名称

类型

描述

item

object

\-

value

object

\-

接口示例

import akshare as ak

stock\_individual\_info\_em\_df \= ak.stock\_individual\_info\_em(symbol\="000001")
print(stock\_individual\_info\_em\_df)

数据示例

   item               value
0    最新               7.05
1  股票代码             000002
2  股票简称            万  科Ａ
3   总股本       11930709471.0
4   流通股        9716935865.0
5   总市值  84111501770.550003
6  流通市值      68504397848.25
7    行业              房地产开发
8  上市时间            19910129

### 个股信息查询-雪球[](https://akshare.akfamily.xyz/data/stock/stock.html#id10 "Link to this heading")

接口: stock\_individual\_basic\_info\_xq

目标地址: https://xueqiu.com/snowman/S/SH601127/detail#/GSJJ

描述: 雪球财经-个股-公司概况-公司简介

限量: 单次返回指定 symbol 的个股信息

输入参数

名称

类型

描述

symbol

str

symbol="SH601127"; 股票代码

token

str

token=None;

timeout

float

timeout=None; 默认不设置超时参数

输出参数

名称

类型

描述

item

object

\-

value

object

\-

接口示例

import akshare as ak

stock\_individual\_basic\_info\_xq\_df \= ak.stock\_individual\_basic\_info\_xq(symbol\="SH601127")
print(stock\_individual\_basic\_info\_xq\_df)

数据示例

                            item                                              value
0                         org\_id                                         T000071215
1                    org\_name\_cn                                        赛力斯集团股份有限公司
2              org\_short\_name\_cn                                                赛力斯
3                    org\_name\_en                               Seres Group Co.,Ltd.
4              org\_short\_name\_en                                              SERES
5        main\_operation\_business      新能源汽车及核心三电(电池、电驱、电控)、传统汽车及核心部件总成的研发、制造、销售及服务。
6                operating\_scope  　　一般项目：制造、销售：汽车零部件、机动车辆零部件、普通机械、电器机械、电器、电子产品（不...
7                district\_encode                                             500106
8            org\_cn\_introduction  赛力斯始创于1986年，是以新能源汽车为核心业务的技术科技型汽车企业。现有员工1.6万人，A...
9           legal\_representative                                                张正萍
10               general\_manager                                                张正萍
11                     secretary                                                 申薇
12              established\_date                                      1178812800000
13                     reg\_asset                                       1509782193.0
14                     staff\_num                                              16102
15                     telephone                                     86-23-65179666
16                      postcode                                             401335
17                           fax                                     86-23-65179777
18                         email                                    601127@seres.cn
19                   org\_website                                   www.seres.com.cn
20                reg\_address\_cn                                      重庆市沙坪坝区五云湖路7号
21                reg\_address\_en                                               None
22             office\_address\_cn                                      重庆市沙坪坝区五云湖路7号
23             office\_address\_en                                               None
24               currency\_encode                                             019001
25                      currency                                                CNY
26                   listed\_date                                      1465920000000
27               provincial\_name                                                重庆市
28             actual\_controller                                       张兴海 (13.79%)
29                   classi\_name                                               民营企业
30                   pre\_name\_cn                                     重庆小康工业集团股份有限公司
31                      chairman                                                张正萍
32               executives\_nums                                                 20
33              actual\_issue\_vol                                        142500000.0
34                   issue\_price                                               5.81
35             actual\_rc\_net\_amt                                        738451000.0
36              pe\_after\_issuing                                              18.19
37  online\_success\_rate\_of\_issue                                           0.110176
38            affiliate\_industry         {'ind\_code': 'BK0025', 'ind\_name': '汽车整车'}