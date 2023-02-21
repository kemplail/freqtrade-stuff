from freqtrade.strategy import stoploss_from_open
from pandas import DataFrame

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
import importlib

interface_futures_binance = importlib.import_module("interface_futures_binance")
importlib.reload(interface_futures_binance)


class SimpleBinanceFutures(interface_futures_binance.IFutures):
    timeframe = '5m'

    # ROI table:
    minimal_roi = {
        "0": 0.065,
        "7": 0.035,
        "18": 0.012,
        "42": 0,
    }

    # Stoploss:
    stoploss = -0.7
    use_custom_stoploss = True

    # Futures config
    # New binance users only support 20x leverage, please confirm whether your account supports higher leverage!
    # Make sure your API is enabled and have Futures permissions
    # (open futures after created the API, the created API still does not support futres, you must create a new API)
    _leverage = 10

    def custom_stoploss(
        self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs
    ) -> float:
        """
        Must! Must! Must use custom_stoploss, and turn on stoploss_on_exchange

        FT spot does not handle the logic of forced liquidation.
        Therefore, stoploss_on_exchange must be used to manually liquidate the position before the forced liquidation
        """
        p = locals()
        del p["self"], p["__class__"]

        return stoploss_from_open(super().custom_stoploss(**p), current_profit)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'buy'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'sell'] = 0

        return dataframe