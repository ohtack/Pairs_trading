import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa import arima_model
import math
from datetime import datetime

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    all_pairs = []
    pairs = []

    # result
    stock1 = []
    stock2 = []
    pvalue_list = []
    check_95 = []
    check_98 = []

    for i in range(n):
        for j in range(i + 1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue

            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
                check_95.append('Y')
            else:
                check_95.append('N')

            if pvalue < 0.02:
                check_98.append('Y')
            else:
                check_98.append('N')

            stock1.append(keys[i])
            stock2.append(keys[j])
            pvalue_list.append(pvalue)

    pair_pvalue = pd.DataFrame()
    pair_pvalue['s1'] = stock1
    pair_pvalue['s2'] = stock2
    pair_pvalue['pvalue'] = pvalue_list
    pair_pvalue['check_95'] = check_95
    pair_pvalue['check_98'] = check_98

    pair_pvalue.sort_values('pvalue', ascending=True, inplace=True)

    return score_matrix, pvalue_matrix, pair_pvalue, pairs


def get_para(stock1, stock2):

    stock1 = stock1.dropna()
    stock2 = stock2.dropna()
    arima = arima_model.ARIMA(stock1, [1, 0, 0], stock2)
    arima_result = arima.fit()
    c = arima_result.params[1]
    z = np.array(stock1) - (c * np.array(stock2))
    mean = np.mean(z)
    std = np.std(z)

    return c, z, mean, std


def get_z(stock1, stock2, c):
    value_z = stock1 - (c * stock2)
    return value_z



def invest_std(stock1, stock2, window):

    log_ret_s1 = stock1[:window]
    log_ret_s2 = stock2[:window]
    log_ret_s1 = np.log(log_ret_s1)-np.log(log_ret_s1.shift(1))
    log_ret_s2 = np.log(log_ret_s2)-np.log(log_ret_s2.shift(1))
    log_ret_s1 = log_ret_s1.dropna()
    log_ret_s2 = log_ret_s1.dropna()

    money = 1000000
    balance = []
    position = 0
    num_stock1 = 0
    num_stock2 = 0

    c, z, mean, std = get_para(log_ret_s1, log_ret_s2)

    # Mean + Standard deviation trade

    for i in range(window, len(stock1)):

        today_s1 = stock1[i]
        today_s2 = stock2[i]
        yes_s1 = stock1[i-1]
        yes_s2 = stock2[i-1]
        today_log1 = np.log(today_s1) - np.log(yes_s1)
        today_log2 = np.log(today_s2) - np.log(yes_s2)

        print('Date: '+str(datetime.strftime(stock1.keys()[i],"%Y-%m-%d")))
        print()
        
        if i == (len(stock1)-1):
            
            print('Closed position (Last day)')
            
            if position == 1:
                
                money = money + (num_stock1 * stock1[i]) - (num_stock2 * stock2[i])
                num_stock1 = 0
                num_stock2 = 0
                position = 0
                balance.append(money)
             
            elif position == 2:
                
                money = money - (num_stock1 * stock1[i]) + (num_stock2 * stock2[i])
                num_stock1 = 0
                num_stock2 = 0
                position = 0
                balance.append(money)
                                
            
            print('Last Balance: {}'.format(money))
            print()
            
            continue

        if position == 0:

            z_today = get_z(today_log1, today_log2, c)

            if z_today <= mean - std:

                num_stock1 = math.floor(money / stock1[i])
                num_stock2 = math.floor((num_stock1 * stock1[i]) * c ) / stock2[i]
                money = money - (num_stock1 * stock1[i]) + (num_stock2 * stock2[i])

                position = 1
                balance.append(money)

                print('long position on '+stock1.name +' and short position on '+ stock2.name)
                print('Money Balance: {}'.format(money))
                print()

            elif z_today >= mean + std:

                num_stock1 = math.floor(money / stock1[i])
                num_stock2 = math.floor((num_stock1 * stock1[i]) * c) / stock2[i]
                money = money + (num_stock1 * stock1[i]) - (num_stock2 * stock2[i])

                position = 2
                balance.append(money)

                print('long position on ' + stock2.name + ' and short position on ' + stock1.name)
                print('Money Balance: {}'.format(money))
                print()

            else:
                position = 0
                balance.append(money)
                
                print('no trading')
                print()



        elif position == 1:

            z_today = get_z(today_log1, today_log2, c)

            if z_today >= mean:

                money = money + (num_stock1 * stock1[i]) - (num_stock2 * stock2[i])
                num_stock1 = 0
                num_stock2 = 0
                position = 0
                balance.append(money)

                print('Closed position')
                print('Money Balance: {}'.format(money))
                print()

            else:

                position = 1
                balance.append(money)
                
                print('Hold position')
                print()


        elif position == 2:

            z_today = get_z(today_log1, today_log2, c)

            if z_today <= mean:
                money = money - (num_stock1 * stock1[i]) + (num_stock2 * stock2[i])
                num_stock1 = 0
                num_stock2 = 0
                position = 0
                balance.append(money)

                print('Closed position')
                print('Money Balance: {}'.format(money))
                print()

            else:
                position = 2
                balance.append(money)
                
                print('Hold position')
                print()
    
    

    rate = (money / 1000000) - 1
    return money, rate, num_stock1, num_stock2, position, balance

# Mean + Standard deviation * 2 trade
def invest_std2(stock1, stock2, window):
    log_ret_s1 = stock1[:window]
    log_ret_s2 = stock2[:window]
    log_ret_s1 = np.log(log_ret_s1) - np.log(log_ret_s1.shift(1))
    log_ret_s2 = np.log(log_ret_s2) - np.log(log_ret_s2.shift(1))
    log_ret_s1 = log_ret_s1.dropna()
    log_ret_s2 = log_ret_s1.dropna()

    money = 1000000
    balance = []
    position = 0
    num_stock1 = 0
    num_stock2 = 0

    c, z, mean, std = get_para(log_ret_s1, log_ret_s2)

    for i in range(window, len(stock1)):

        today_s1 = stock1[i]
        today_s2 = stock2[i]
        yes_s1 = stock1[i - 1]
        yes_s2 = stock2[i - 1]
        today_log1 = np.log(today_s1) - np.log(yes_s1)
        today_log2 = np.log(today_s2) - np.log(yes_s2)

        print('Date: ' + str(datetime.strftime(stock1.keys()[i], "%Y-%m-%d")))
        print()

        if i == (len(stock1) - 1):

            print('Closed position (Last day)')

            if position == 1:

                money = money + (num_stock1 * stock1[i]) - (num_stock2 * stock2[i])
                num_stock1 = 0
                num_stock2 = 0
                position = 0
                balance.append(money)

            elif position == 2:

                money = money - (num_stock1 * stock1[i]) + (num_stock2 * stock2[i])
                num_stock1 = 0
                num_stock2 = 0
                position = 0
                balance.append(money)

            print('Last Balance: {}'.format(money))
            print()

            continue

        if position == 0:

            z_today = get_z(today_log1, today_log2, c)

            if z_today <= mean - (2 * std):

                num_stock1 = math.floor(money / stock1[i])
                num_stock2 = math.floor((num_stock1 * stock1[i]) * c) / stock2[i]
                money = money - (num_stock1 * stock1[i]) + (num_stock2 * stock2[i])

                position = 1
                balance.append(money)

                print('long position on ' + stock1.name + ' and short position on ' + stock2.name)
                print('Money Balance: {}'.format(money))
                print()

            elif z_today >= mean + (2 * std):

                num_stock1 = math.floor(money / stock1[i])
                num_stock2 = math.floor((num_stock1 * stock1[i]) * c) / stock2[i]
                money = money + (num_stock1 * stock1[i]) - (num_stock2 * stock2[i])

                position = 2
                balance.append(money)

                print('long position on ' + stock2.name + ' and short position on ' + stock1.name)
                print('Money Balance: {}'.format(money))
                print()

            else:
                position = 0
                balance.append(money)

                print('no trading')
                print()



        elif position == 1:

            z_today = get_z(today_log1, today_log2, c)

            if z_today >= mean:

                money = money + (num_stock1 * stock1[i]) - (num_stock2 * stock2[i])
                num_stock1 = 0
                num_stock2 = 0
                position = 0
                balance.append(money)

                print('Closed position')
                print('Money Balance: {}'.format(money))
                print()

            else:

                position = 1
                balance.append(money)

                print('Hold position')
                print()


        elif position == 2:

            z_today = get_z(today_log1, today_log2, c)

            if z_today <= mean:
                money = money - (num_stock1 * stock1[i]) + (num_stock2 * stock2[i])
                num_stock1 = 0
                num_stock2 = 0
                position = 0
                balance.append(money)

                print('Closed position')
                print('Money Balance: {}'.format(money))
                print()

            else:
                position = 2
                balance.append(money)

                print('Hold position')
                print()

    rate = (money / 1000000) - 1
    return money, rate, num_stock1, num_stock2, position, balance
