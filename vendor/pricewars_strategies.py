"""Price Wars reference strategies (MIT), vendored without network dependencies.
Source: https://github.com/hpi-epic/pricewars-merchant/blob/7981dad52034493f1f77e21e2aa297598c6ef769/merchant.py
"""

class CheapestStrategy:
    name = 'cheapest'

    settings = {
        'price_decrement': 0.05,
        'default_price': 30
    }

    # set color used in graphs for this merchant, if not set it will be a random color
    # e31a1c = red
    color = "#e31a1c"

    @staticmethod
    def calculate_price(merchant, offer_id, market_situation):
        product_id = [offer for offer in market_situation if offer.offer_id == offer_id][0].product_id
        relevant_competitor_offers = [offer for offer in market_situation if
                                      offer.product_id == product_id and
                                      offer.merchant_id != merchant.merchant_id]
        if not relevant_competitor_offers:
            return merchant.settings['default_price']

        cheapest_offer = min(relevant_competitor_offers, key=lambda offer: offer.price)
        return cheapest_offer.price - merchant.settings['price_decrement']

class TwoBoundStrategy:
    name = 'two_bound'

    settings = {
        'price_decrement': 0.10,
        'upper_price_bound': 30,
        'lower_price_bound': 20
    }

    # 33a02c = green
    color = "#33a02c"

    @staticmethod
    def calculate_price(merchant, offer_id, market_situation):
        product_id = [offer for offer in market_situation if offer.offer_id == offer_id][0].product_id
        relevant_competitor_offers = [offer for offer in market_situation if
                                      offer.product_id == product_id and
                                      offer.merchant_id != merchant.merchant_id]
        if not relevant_competitor_offers:
            return merchant.settings['upper_price_bound']

        cheapest_offer = min(relevant_competitor_offers, key=lambda offer: offer.price)
        if cheapest_offer.price <= merchant.settings['lower_price_bound'] or \
                cheapest_offer.price > merchant.settings['upper_price_bound']:
            return merchant.settings['upper_price_bound']
        else:
            return cheapest_offer.price - merchant.settings['price_decrement']
