from functools import reduce
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class FUTURES(IStrategy):

    INTERFACE_VERSION = 3
    timeframe = "1m"
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {"0": 0.13}
    # minimal_roi = {"0": 1}

    stoploss = -0.2
    can_short = True

    # Trailing stoploss
    trailing_stop  = True
    trailing_only_offset_is_reached =  True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.05 # Disabled / not configured


    # Run "populate_indicators()" only for new candle.
   # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 5
    use_exit_signal = False

    # Hyperoptable parameters

    # Define the guards spaces


    # Define the parameter spaces

    def leverage(self, pair: str, current_time: 'datetime', current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return  5.0
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema34'] = ta.EMA(dataframe, timeperiod=34)
        dataframe['ema55'] = ta.EMA(dataframe,timeperiod=55)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['ema5'],dataframe['ema8'])) &
                (qtpylib.crossed_above(dataframe['ema5'],dataframe['ema13'])) &
               (qtpylib.crossed_above(dataframe['ema5'],dataframe['ema21'])) &
              # (qtpylib.crossed_above(dataframe['ema5'],dataframe['ema34'])) &
                (dataframe['ema5']>dataframe['ema8'])&
                (dataframe['volume'] > 0)
            ),
            "enter_long",
        ] = 1

        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['ema5'],dataframe['ema8'])) &
                 (qtpylib.crossed_below(dataframe['ema5'],dataframe['ema13'])) &
                 (qtpylib.crossed_below(dataframe['ema5'],dataframe['ema21'])) &
                 # (qtpylib.crossed_below(dataframe['ema5'],dataframe['ema34'])) &
                   (dataframe['ema5']<dataframe['ema8'])&
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            "enter_short",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:



        dataframe.loc[
              (
                (qtpylib.crossed_below(dataframe['ema5'], dataframe['ema8'])) &
                (qtpylib.crossed_below(dataframe['ema5'],dataframe['ema13']))&
                 (qtpylib.crossed_below(dataframe['ema5'],dataframe['ema21']))
            ),
            "exit_long",
        ] = 1

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['ema5'], dataframe['ema8'])) &
                (qtpylib.crossed_above(dataframe['ema5'],dataframe['ema13']))&
                 (qtpylib.crossed_above(dataframe['ema5'],dataframe['ema21']))
            ),
            "exit_short",
        ] = 1

        return dataframe