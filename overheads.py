"""Daily costs and laws outside customer flow: wages, moving, and election laws."""
from mechanics import food_profit
from politics import law_terms, law_profit, charged_price


class Overheads:
    def __init__(self, state, spot, salvage):
        flags = state.get('features') or {}
        you = state.get('you') or {}
        self.salvage = salvage
        self.fixed = 0.
        self.wage, self.per, self.capacity = 0., 50, None
        # Coupons dropped on us and not yet used: each puts 20% of our buyers at half price (stacking to 60%).
        pending = sum(1 for c in (you.get('coupons') or []) if isinstance(c, dict) and c.get('applied_day') is None)
        self.coupon_factor = 1. - .5 * min(.6, .2 * pending)
        if flags.get('staff') and isinstance(you.get('staff'), dict):
            crew = you['staff']
            staff_rules = (state.get('rules') or {}).get('staff') or {}
            self.per = max(1, int(staff_rules.get('meals_per_employee') or 50))
            self.wage = float(crew.get('wage', staff_rules.get('default_wage', 0)) or 0)
            # Tomorrow's crew is set tonight, so today's crew is not a limit on tomorrow's order: wages follow
            # the meals cooked (one employee per 50) and only the game's headcount maximum bounds the order.
            self.capacity = int(staff_rules.get('staff_max') or 10) * self.per
        if flags.get('moving_costs') and you.get('last_spot') not in (None, spot):
            table = (state.get('rules') or {}).get('move_costs') or {}
            self.fixed += float((table.get(you['last_spot']) or {}).get(spot, 0))
        laws = (state.get('politics') or {}).get('laws') if flags.get('elections') else None
        self.terms = law_terms(laws)

    def charged(self, price):
        return charged_price(self.terms, price)

    def profit(self, sales, meals, price, quality, cost):
        """Profit after wages, moving and laws. Opening costs nothing extra with zero meals."""
        if not meals:
            return 0.
        charged = self.charged(price)
        wages = self.wage * max(1, -(-int(meals) // self.per))  # employees needed for these meals
        return (food_profit(sales, meals, charged * self.coupon_factor, cost, self.salvage)
                + law_profit(self.terms, sales, charged, quality) - self.fixed - wages)
