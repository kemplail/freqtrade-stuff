import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class LeoStrategy(IStrategy):

    INTERFACE_VERSION = 3

    can_short: bool = False

    minimal_roi = {
        "90": 0.01,
        "60": 0.02,
        "30": 0.05,
        "0": 0.1
    }

    stoploss = -0.14
    trailing_stop = True
    trailing_stop_positive = 0.07
    
    timeframe = '4h'

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 60

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # Momentum Indicators
        # ------------------------------------

        # EMA 10
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)

        # EMA 60
        dataframe['ema60'] = ta.EMA(dataframe, timeperiod=60)

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['ema10'], dataframe['ema60'])) &
                (dataframe['ema10'] > dataframe['ema10'].shift(1)) & 
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['ema10'], dataframe['ema60'])) &
                (dataframe['ema10'] < dataframe['ema10'].shift(1)) & 
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 1

        return dataframe
