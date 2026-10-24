"""Daily costs and laws outside customer flow: wages, moving, and election laws."""
from mechanics import food_profit
from politics import law_terms, law_profit, charged_price


class Overheads:
    def __init__(self, state, spot, salvage):
        flags = state.get('features') or {}
        you = state.get('you') or {}
        self.salvage = salvage
        self.fixed = 0.
        self.capacity = None
        if flags.get('staff') and isinstance(you.get('staff'), dict):
            crew = you['staff']
            employees = crew.get('employees', crew.get('target', 0)) or 0
            self.fixed += float(employees) * float(crew.get('wage', 0) or 0)
            if crew.get('capacity') is not None:
                self.capacity = int(crew['capacity'])
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
        return (food_profit(sales, meals, charged, cost, self.salvage)
                + law_profit(self.terms, sales, charged, quality) - self.fixed)
