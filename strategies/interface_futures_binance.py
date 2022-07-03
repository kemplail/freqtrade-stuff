import ccxt
import arrow
import logging
from freqtrade.exchange import Exchange, Binance
from freqtrade.wallets import Wallets, Wallet
from freqtrade.persistence import LocalTrade, Trade
from freqtrade.exchange.common import retrier
from freqtrade.exceptions import (
    DDosProtection, InsufficientFundsError, ExchangeError, InvalidOrderException, TemporaryError, OperationalException
)
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import stoploss_from_open
from datetime import datetime, timedelta
from freqtrade.enums import RunMode
from freqtrade.constants import UNLIMITED_STAKE_AMOUNT
from typing import Dict, List, Optional, Tuple, Union, Any
from pandas import DataFrame

logger = logging.getLogger(__name__)


class IFutures(IStrategy):
    _initialized_futures: bool = False
    _leverage: int = 1
    _isolated: bool = True
    _maintenance_stoploss: Dict = {}

    def custom_stake_amount(
        self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float, min_stake: float,
        max_stake: float, **kwargs
    ) -> float:

        if self.config['stake_amount'] == UNLIMITED_STAKE_AMOUNT:
            return proposed_stake
        else:
            return proposed_stake * self._leverage

    def custom_stoploss(
        self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs
    ) -> float:
        p = locals()
        del p["self"], p["__class__"]
        stoploss = super().custom_stoploss(**p)

        if pair in self._maintenance_stoploss:
            stoploss = self._maintenance_stoploss[pair]

        return stoploss

    def confirm_trade_entry(
        self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, current_time: datetime,
        **kwargs
    ) -> bool:

        if self.dp.runmode not in [RunMode.LIVE]:
            self._maintenance_stoploss[pair] = self.stoploss

        if pair not in self._maintenance_stoploss:
            self.reset_leverage()

        return pair in self._maintenance_stoploss

    def ohlcvdata_to_dataframe(self, data: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        # in hyper not call bot_loop_start.so....
        self.prepare_futures()

        p = locals()
        del p["self"], p["__class__"]
        return super().ohlcvdata_to_dataframe(**p)

    def bot_loop_start(self, **kwargs) -> None:
        self.prepare_futures()

    def prepare_futures(self):
        if self._initialized_futures:
            return

        self.check_config()

        self.hook_method()

        self.reset_stoploss()

        if self.dp.runmode in [RunMode.LIVE]:
            self.reset_leverage()

        self._initialized_futures = True

    def check_config(self):
        if self._leverage < 1:
            raise OperationalException(f'Leverage must be grater than 0')

        if "edge" in self.config and "enabled" in self.config["edge"] and self.config["edge"]["enabled"]:
            raise OperationalException('Futures does not support edge')

        if self.config["exchange"]["name"] != "binance":
            raise OperationalException(f'Future only supports binance')

    def hook_method(self):
        Exchange.get_balances = get_balances
        Exchange.leverage = self._leverage
        Exchange.isolated = self._isolated
        Binance._params = {"workingType": "MARK_PRICE"}
        Binance.stoploss_adjust = stoploss_adjust
        Binance.stoploss = stoploss
        Binance.create_dry_run_order = create_dry_run_order
        Wallets.get_free = get_free
        Wallets.get_total_stake_amount = get_total_stake_amount
        Wallets._update_dry = _update_dry
        Wallets.leverage = self._leverage
        Wallets.isolated = self._isolated
        logger.info('Hook spot method!')

    def reset_stoploss(self):
        default_maintenance = 0 if self.dp.runmode in [RunMode.LIVE] else 0.01
        self.stoploss = max(self.stoploss, -1 / self._leverage + default_maintenance)
        logger.info(f'Reset stoploss to {self.stoploss}!')

    def reset_leverage(self):
        logger.info(f'Reset leverage and isolated and dual side!')

        if self.wallets._exchange._api.fapiPrivateGetPositionsideDual()["dualSidePosition"]:
            self.wallets._exchange._api.fapiPrivateGetPositionsideDual(params={"dualSidePosition": False})

        for x in self.wallets._exchange._api.fetch_positions():
            self._maintenance_stoploss[x["symbol"]] = max(
                -1 / self._leverage + x["maintenanceMarginPercentage"],
                self.stoploss,
            )

        for x in self.wallets._exchange._api.fetch_balance()["info"]["positions"]:
            if int(x["leverage"]) != self._leverage:
                logger.info('set leverage %s. leverage: %s.', x["symbol"], self._leverage)
                self.wallets._exchange._api.set_leverage(
                    symbol=self.wallets._exchange._api.markets_by_id[x["symbol"]]["symbol"],
                    leverage=self._leverage,
                )
            if x["isolated"] != self._isolated:
                logger.info('set isolated %s. isolated: %s.', x["symbol"], "ISOLATED" if self._isolated else "CROSSED")
                self.wallets._exchange._api.set_margin_mode(
                    symbol=self.wallets._exchange._api.markets_by_id[x["symbol"]]["symbol"],
                    marginType="ISOLATED" if self._isolated else "CROSSED",
                )

        self.wallets._exchange.reload_markets()


# ------------------------------------------------------------------------
def get_free(self, currency: str) -> float:
    leverage = self.leverage if currency == self._config['stake_currency'] else 1

    balance = self._wallets.get(currency)
    if balance and balance.free:
        return balance.free * leverage
    else:
        return 0

def get_total_stake_amount(self):
    val_tied_up = Trade.total_open_trades_stakes()
    if "available_capital" in self._config:
        starting_balance = self._config['available_capital']
        disabled_balance = self._config.get('disabled_capital', 0)
        starting_balance = starting_balance - disabled_balance
        tot_profit = Trade.get_total_closed_profit()
        available_amount = (starting_balance + tot_profit) * self.leverage * self._config['tradable_balance_ratio']

    else:
        available_amount = ((val_tied_up + self.get_free(self._config['stake_currency'])) *
                            self._config['tradable_balance_ratio'])
    return available_amount


def _update_dry(self) -> None:

    _wallets = {}
    open_trades = Trade.get_trades_proxy(is_open=True)
    # If not backtesting...
    # TODO: potentially remove the ._log workaround to determine backtest mode.
    if self._log:
        tot_profit = Trade.get_total_closed_profit()
    else:
        tot_profit = LocalTrade.total_profit
    tot_in_trades = sum([trade.stake_amount for trade in open_trades])

    current_stake = self.start_cap + tot_profit - tot_in_trades / self.leverage
    _wallets[self._config['stake_currency']] = Wallet(self._config['stake_currency'], current_stake, 0, current_stake)

    for trade in open_trades:
        curr = self._exchange.get_pair_base_currency(trade.pair)
        _wallets[curr] = Wallet(curr, trade.amount, 0, trade.amount)
    self._wallets = _wallets


@retrier
def get_balances(self) -> dict:

    try:
        balances = self._api.fetch_balance()
        if "positions" in balances["info"]:
            for x in balances["info"]["positions"]:
                if x["symbol"] not in self._api.markets_by_id:
                    continue

                quote = self._api.markets_by_id[x["symbol"]]["quote"]
                if quote != self._config["stake_currency"]:
                    continue

                if float(x["positionAmt"]) <= 0:
                    continue

                base = self._api.markets_by_id[x["symbol"]]["base"]
                balances[base] = {"free": float(x["positionAmt"]), "used": 0, "total": float(x["positionAmt"])}

        # Remove additional info from ccxt results
        balances.pop("info", None)
        balances.pop("free", None)
        balances.pop("total", None)
        balances.pop("used", None)

        return balances
    except ccxt.DDoSProtection as e:
        raise DDosProtection(e) from e
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        raise TemporaryError(f'Could not get balance due to {e.__class__.__name__}. Message: {e}') from e
    except ccxt.BaseError as e:
        raise OperationalException(e) from e


def stoploss_adjust(self, stop_loss: float, order: Dict) -> bool:
    return order['type'] == 'stop' and stop_loss > float(order['info']['stopPrice'])


@retrier(retries=0)
def stoploss(self, pair: str, amount: float, stop_price: float, order_types: Dict) -> Dict:
    """
    creates a stoploss limit order.
    this stoploss-limit is binance-specific.
    It may work with a limited number of other exchanges, but this has not been tested yet.
    """
    # Limit price threshold: As limit price should always be below stop-price
    limit_price_pct = order_types.get('stoploss_on_exchange_limit_ratio', 0.99)
    rate = stop_price * limit_price_pct

    ordertype = "stop"

    stop_price = self.price_to_precision(pair, stop_price)

    # Ensure rate is less than stop price
    if stop_price <= rate:
        raise OperationalException('In stoploss limit order, stop price should be more than limit price')

    if self._config['dry_run']:
        dry_order = self.create_dry_run_order(pair, ordertype, "sell", amount, stop_price)
        return dry_order

    try:
        params = self._params.copy()
        params.update({'stopPrice': stop_price})

        amount = self.amount_to_precision(pair, amount)

        rate = self.price_to_precision(pair, rate)

        order = self._api.create_order(
            symbol=pair, type=ordertype, side='sell', amount=amount, price=rate, params=params
        )
        logger.info('stoploss limit order added for %s. ' 'stop price: %s. limit: %s', pair, stop_price, rate)
        self._log_exchange_response('create_stoploss_order', order)
        return order
    except ccxt.InsufficientFunds as e:
        raise InsufficientFundsError(
            f'Insufficient funds to create {ordertype} sell order on market {pair}. '
            f'Tried to sell amount {amount} at rate {rate}. '
            f'Message: {e}'
        ) from e
    except ccxt.InvalidOrder as e:
        # Errors:
        # `binance Order would trigger immediately.`
        raise InvalidOrderException(
            f'Could not create {ordertype} sell order on market {pair}. '
            f'Tried to sell amount {amount} at rate {rate}. '
            f'Message: {e}'
        ) from e
    except ccxt.DDoSProtection as e:
        raise DDosProtection(e) from e
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        raise TemporaryError(f'Could not place sell order due to {e.__class__.__name__}. Message: {e}') from e
    except ccxt.BaseError as e:
        raise OperationalException(e) from e


def create_dry_run_order(self,
                         pair: str,
                         ordertype: str,
                         side: str,
                         amount: float,
                         rate: float,
                         params: Dict = {}) -> Dict[str, Any]:
    order_id = f'dry_run_{side}_{datetime.now().timestamp()}'
    _amount = self.amount_to_precision(pair, amount)
    dry_order: Dict[str, Any] = {
        'id': order_id,
        'symbol': pair,
        'price': rate,
        'average': rate,
        'amount': _amount,
        'cost': _amount * rate,
        'type': ordertype,
        'side': side,
        'remaining': _amount,
        'datetime': arrow.utcnow().isoformat(),
        'timestamp': arrow.utcnow().int_timestamp * 1000,
        'status': "closed" if ordertype == "market" else "open",
        'fee': None,
        'info': {}
    }
    if dry_order["type"] in ["stop", "stop_loss_limit", "stop-loss-limit"]:
        dry_order["info"] = {"stopPrice": dry_order["price"]}

    if dry_order["type"] == "market":
        # Update market order pricing
        average = self.get_dry_market_fill_price(pair, side, amount, rate)
        dry_order.update({
            'average': average,
            'cost': dry_order['amount'] * average,
        })
        dry_order = self.add_dry_order_fee(pair, dry_order)

    dry_order = self.check_dry_limit_order_filled(dry_order)

    self._dry_run_open_orders[dry_order["id"]] = dry_order
    # Copy order and close it - so the returned order is open unless it's a market order
    return dry_order