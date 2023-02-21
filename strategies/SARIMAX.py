import statsmodels.api as sm
import custom_indicators as cta
import warnings
import logging
from pathlib import Path
import sys
import numpy as np
import scipy.fft
from scipy.fft import rfft, irfft
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow

from freqtrade.strategy import (IStrategy, merge_informative_pair, stoploss_from_open,
                                IntParameter, DecimalParameter, CategoricalParameter)

from typing import Dict, List, Optional, Tuple, Union
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.persistence import Trade

# Get rid of pandas warnings during backtesting
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy

sys.path.append(str(Path(__file__).parent))


log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


"""
####################################################################################
SARIMAX - use a SARIMAX Filter to estimate future price movements

####################################################################################
"""


class SARIMAX(IStrategy):
    # Do *not* hyperopt for the roi and stoploss spaces

    # ROI table:
    minimal_roi = {
        "0": 10
    }

    # Stoploss:
    stoploss = -0.10

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    timeframe = '5m'
    inf_timeframe = '1h'

    use_custom_stoploss = True

    # Recommended
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Required
    startup_candle_count: int = 32
    process_only_new_candles = True

    custom_trade_info = {}

    ###################################

    # Strategy Specific Variable Storage

    smax_window = startup_candle_count
    filter_list = {}
    filter_init_list = {}
    current_pair = ""

    # Hyperopt Variables

    # SARIMAX  hyperparams
    entry_smax_dif = 4.5
    cexit_endtrend_respect_roi = True
    cexit_pullback = True
    cexit_pullback_amount = 0.007
    cexit_pullback_respect_roi = False
    cexit_roi_end = 0.001
    cexit_roi_start = 0.017
    cexit_roi_time = 1420
    cexit_roi_type = "static"
    cexit_trend_type = "none"
    cstop_bail_how = "none"
    cstop_bail_roc = -1.014
    cstop_bail_time = 908
    cstop_bail_time_trend = False
    cstop_loss_threshold = -0.02
    cstop_max_stoploss = -0.13
    exit_smax_diff = -3.3

    ###################################

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        return informative_pairs

    ###################################

    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Base pair informative timeframe indicators
        curr_pair = metadata['pair']
        informative = self.dp.get_pair_dataframe(pair=curr_pair, timeframe=self.inf_timeframe)

        # SARIMAX Filter

        # get filter for current pair

        self.current_pair = curr_pair

        # create if not already done
        if not curr_pair in self.filter_list:
            self.filter_init_list[curr_pair] = False

        informative['smax_predict'] = informative['close'].rolling(
            window=self.smax_window).apply(self.model)

        # merge into normal timeframe
        dataframe = merge_informative_pair(
            dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        # calculate predictive indicators in shorter timeframe (not informative)

        dataframe['smax_predict'] = dataframe[f"smax_predict_{self.inf_timeframe}"]
        dataframe['smax_predict_diff'] = 100.0 * \
            (dataframe['smax_predict'] - dataframe['close']) / dataframe['close']

        # Custom Stoploss

        if not metadata['pair'] in self.custom_trade_info:
            self.custom_trade_info[metadata['pair']] = {}
            if not 'had-trend' in self.custom_trade_info[metadata["pair"]]:
                self.custom_trade_info[metadata['pair']]['had-trend'] = False

        # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        dataframe['rmi'] = cta.RMI(dataframe, length=24, mom=5)

        # MA Streak: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
        dataframe['mastreak'] = cta.mastreak(dataframe, period=4)

        # Trends, Peaks and Crosses
        dataframe['candle-up'] = np.where(dataframe['close'] >= dataframe['open'], 1, 0)
        dataframe['candle-up-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() >= 3, 1, 0)

        dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(), 1, 0)
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() >= 3, 1, 0)

        dataframe['rmi-dn'] = np.where(dataframe['rmi'] <= dataframe['rmi'].shift(), 1, 0)
        dataframe['rmi-dn-count'] = dataframe['rmi-dn'].rolling(8).sum()

        # Indicators used only for ROI and Custom Stoploss
        ssldown, sslup = cta.SSLChannels_ATR(dataframe, length=21)
        dataframe['sroc'] = cta.SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['ssl-dir'] = np.where(sslup > ssldown, 'up', 'down')

        return dataframe

    ###################################

    def model(self, a: np.ndarray) -> np.float:

        # scale the data
        standardized = a.copy()
        w_mean = np.mean(standardized)
        w_std = np.std(standardized)
        scaled = (standardized - w_mean) / w_std
        scaled.fillna(0, inplace=True)

        # create filter if needed
        # if not self.filter_init_list[self.current_pair]:
        #     self.filter_init_list[self.current_pair] = True
        # fit an AR(2) model
        s_model = sm.tsa.SARIMAX(
            a, order=(2, 0, 0), enforce_invertibility=False, enforce_stationarity=False)
        # print("params: ", s_model.params_complete)

        result = s_model.fit(disp=False)
        # print(result.summary())
        # print("retvals: ", result.mle_retvals)
        # print("result: ", result)

        # get the prediction
        forecast = result.forecast(2)
        predict = [column for column in forecast]
        # print("predict: ", predict)

        length = len(predict)
        return (predict[length - 1] * w_std) + w_mean

    # ###################################
    #
    # # Williams %R
    # def williams_r(self, dataframe: DataFrame, period: int = 14) -> Series:
    #     """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
    #         of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
    #         Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
    #         of its recent trading range.
    #         The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    #     """
    #
    #     highest_high = dataframe["high"].rolling(center=False, window=period).max()
    #     lowest_low = dataframe["low"].rolling(center=False, window=period).min()
    #
    #     WR = Series(
    #         (highest_high - dataframe["close"]) / (highest_high - lowest_low),
    #         name=f"{period} Williams %R",
    #     )
    #
    #     return WR * -100
    ###################################

    """
    Buy Signal
    """

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        # conditions.append(dataframe['volume'] > 0)

        # SARIMAX triggers
        smax_cond = (
                qtpylib.crossed_above(dataframe['smax_predict_diff'], self.entry_smax_diff)
        )

        conditions.append(smax_cond)

        # Model will spike on big gains, so try to constrain
        spike_cond = (
                dataframe['smax_predict_diff'] < 2.0 * self.entry_smax_diff
        )
        conditions.append(spike_cond)

        # set buy tags
        dataframe.loc[smax_cond, 'enter_tag'] += 'smax_buy '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1

        return dataframe

    ###################################

    """
    Sell Signal
    """

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''

        # SARIMAX triggers
        smax_cond = (
                qtpylib.crossed_below(dataframe['smax_predict_diff'], self.exit_smax_diff)
        )

        conditions.append(smax_cond)

        # DWTs will spike on big gains, so try to constrain
        spike_cond = (
                dataframe['smax_predict_diff'] > 2.0 * self.exit_smax_diff
        )
        conditions.append(spike_cond)

        # set sell tags
        dataframe.loc[smax_cond, 'exit_tag'] += 'smax_sell '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'exit_long'] = 1

        return dataframe

    ###################################

    """
    Custom Stoploss
    """

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        in_trend = self.custom_trade_info[trade.pair]['had-trend']

        # limit stoploss
        if current_profit < self.cstop_max_stoploss:
            return 0.01

        # Determine how we sell when we are in a loss
        if current_profit < self.cstop_loss_threshold:
            if self.cstop_bail_how == 'roc' or self.cstop_bail_how == 'any':
                # Dynamic bailout based on rate of change
                if last_candle['sroc'] <= self.cstop_bail_roc:
                    return 0.01
            if self.cstop_bail_how == 'time' or self.cstop_bail_how == 'any':
                # Dynamic bailout based on time, unless time_trend is True and there is a potential reversal
                if trade_dur > self.cstop_bail_time:
                    if self.cstop_bail_time_trend == True and in_trend == True:
                        return 1
                    else:
                        return 0.01
        return 1

    ###################################

    """
    Custom Sell
    """

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        max_profit = max(0, trade.calc_profit_ratio(trade.max_rate))
        pullback_value = max(0, (max_profit - self.cexit_pullback_amount))
        in_trend = False

        # Determine our current ROI point based on the defined type
        if self.cexit_roi_type == 'static':
            min_roi = self.cexit_roi_start
        elif self.cexit_roi_type == 'decay':
            min_roi = cta.linear_decay(self.cexit_roi_start, self.cexit_roi_end, 0,
                                       self.cexit_roi_time, trade_dur)
        elif self.cexit_roi_type == 'step':
            if trade_dur < self.cexit_roi_time:
                min_roi = self.cexit_roi_start
            else:
                min_roi = self.cexit_roi_end

        # Determine if there is a trend
        if self.cexit_trend_type == 'rmi' or self.cexit_trend_type == 'any':
            if last_candle['rmi-up-trend'] == 1:
                in_trend = True
        if self.cexit_trend_type == 'ssl' or self.cexit_trend_type == 'any':
            if last_candle['ssl-dir'] == 'up':
                in_trend = True
        if self.cexit_trend_type == 'candle' or self.cexit_trend_type == 'any':
            if last_candle['candle-up-trend'] == 1:
                in_trend = True

        # Don't sell if we are in a trend unless the pullback threshold is met
        if in_trend == True and current_profit > 0:
            # Record that we were in a trend for this trade/pair for a more useful sell message later
            self.custom_trade_info[trade.pair]['had-trend'] = True
            # If pullback is enabled and profit has pulled back allow a sell, maybe
            if self.cexit_pullback == True and (current_profit <= pullback_value):
                if self.cexit_pullback_respect_roi == True and current_profit > min_roi:
                    return 'intrend_pullback_roi'
                elif self.cexit_pullback_respect_roi == False:
                    if current_profit > min_roi:
                        return 'intrend_pullback_roi'
                    else:
                        return 'intrend_pullback_noroi'
            # We are in a trend and pullback is disabled or has not happened or various criteria were not met, hold
            return None
        # If we are not in a trend, just use the roi value
        elif in_trend == False:
            if self.custom_trade_info[trade.pair]['had-trend']:
                if current_profit > min_roi:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'trend_roi'
                elif self.cexit_endtrend_respect_roi == False:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'trend_noroi'
            elif current_profit > min_roi:
                return 'notrend_roi'
        else:
            return None
