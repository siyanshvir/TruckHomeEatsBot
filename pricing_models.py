"""Adapt the vendored Price Wars reference merchants to Night Shift price forecasts."""
from types import SimpleNamespace
from vendor.pricewars_strategies import CheapestStrategy, TwoBoundStrategy


def reference_prices(competitor_prices, default, floor, upper):
    offers = [SimpleNamespace(offer_id='ours', product_id='meal', merchant_id='rival', price=default)]
    offers += [SimpleNamespace(offer_id=str(i), product_id='meal', merchant_id='competitor', price=price)
               for i, price in enumerate(competitor_prices)]
    settings = dict(CheapestStrategy.settings, default_price=default)
    cheapest = CheapestStrategy.calculate_price(SimpleNamespace(merchant_id='rival', settings=settings), 'ours', offers)
    settings = dict(TwoBoundStrategy.settings, lower_price_bound=floor, upper_price_bound=upper)
    cycle = TwoBoundStrategy.calculate_price(SimpleNamespace(merchant_id='rival', settings=settings), 'ours', offers)
    # Night Shift requires a food-cost floor for this rival hypothesis. The actual
    # optimizer remains free to price independently and pursue overflow.
    return max(floor, cheapest), max(floor, cycle)
