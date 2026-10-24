#!/usr/bin/env python3
"""Live-data analysis for the Night Shift bot (research stream R1).

Read-only over DATA_ROOT; prints markdown tables to stdout. Standard library only.
Usage: python3 tools/research_live_analysis.py --data-root PATH [--section N]

Data conventions (verified on live market 1):
  * observation file N has last_day = results of day N; its `trucks` list holds
    reputations AFTER day N (so reputation entering day N = file N-1, else 50).
  * spot.customers = arrivals before walking; effective demand shown to the
    spot's trucks = customers + walked_in - walked_out;
    unserved ("hungry") = customers + walked_in - walked_out - total sold.
  * salvage = 25% of ingredient cost of unsold meals; wages = employees * wage.
  * forecast in file N is for day N+1.
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

ME = 't_d64e26'
QV = {'cheap': 8., 'standard': 12., 'premium': 16.}
COST = {'cheap': 2.5, 'standard': 3.75, 'premium': 5.5}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float('nan')


def f(x, n=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '-'
    return ('%.' + str(n) + 'f') % x


def table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join(['---'] * len(head)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    print('\n'.join(out))
    print()


def load(root):
    obs = {}
    for p in glob.glob(os.path.join(root, 'live-observations', '*.json')):
        try:
            x = json.load(open(p))
        except Exception:
            continue
        ld = x.get('last_day') or {}
        d = ld.get('day', x.get('day'))
        if d is not None and ld.get('spots'):
            obs[int(d)] = x
    ev = {}
    pe = os.path.join(root, 'evaluation-v2.jsonl')
    if os.path.exists(pe):
        for line in open(pe):
            try:
                r = json.loads(line)
                ev[int(r['day'])] = r
            except Exception:
                pass
    return obs, ev


def build(obs):
    """One row per truck-day."""
    rows = []
    days = sorted(obs)
    for d in days:
        x = obs[d]
        ld = x['last_day']
        prev = obs.get(d - 1)
        reps = {}
        if prev:
            for t in prev.get('trucks', []):
                if isinstance(t, dict):
                    reps[t['id']] = t.get('reputation') or {}
            if (prev.get('you') or {}).get('reputation'):
                reps[ME] = prev['you']['reputation']
        names = {t['id']: t.get('truck_name') for t in x.get('trucks', []) if isinstance(t, dict)}
        fc = (prev or {}).get('forecast') or {}
        fexp = fc.get('expected_customers') if fc.get('day') == d else None
        cr = (x.get('rules') or {}).get('costs', COST)
        tags = defaultdict(set)
        for m in (ld.get('moments') or []):
            if isinstance(m, dict) and m.get('spot') and m.get('kind') in ('firedrill', 'giveaway', 'protest', 'riot', 'strike'):
                tags[m['spot']].add(m['kind'])
        for spot, s in ld['spots'].items():
            wi, wo = s.get('walked_in', 0) or 0, s.get('walked_out', 0) or 0
            eff = s['customers'] + wi - wo
            tr = s['trucks']
            tot_sold = sum(t['sold'] for t in tr)
            for t in tr:
                rep = (reps.get(t['id']) or {}).get(spot, 50.0)
                surplus = QV[t['quality']] * (0.7 + 0.006 * rep) - t['price']
                rows.append(dict(
                    day=d, weather=ld.get('weather'), events=[(e.get('key') if isinstance(e, dict) else str(e)) for e in ld.get('events', [])],
                    spot=spot, id=t['id'], name=t.get('truck_name') or names.get(t['id']),
                    q=t['quality'], price=t['price'], meals=t['meals'], sold=t['sold'],
                    emp=t.get('employees'), cust=s['customers'], wi=wi, wo=wo, eff=eff,
                    hungry=s.get('hungry'), ff=s.get('free_food'), n=len(tr), tot_sold=tot_sold, rep=rep, surplus=surplus,
                    fexp=(fexp or {}).get(spot), cost_u=cr[t['quality']],
                    tag=','.join(sorted(tags[spot])), sellout=t['sold'] >= t['meals'], me=(t['id'] == ME)))
    return rows


def wage_of(obs, d, tid):
    for t in obs[d].get('trucks', []):
        if isinstance(t, dict) and t['id'] == tid and 'wage' in t:
            return t['wage']
    return 6.0


def est_profit(r, obs, move_prev):
    """Rival profit estimate: revenue - ingredient + salvage - wages - move. (no taxes/fines)"""
    if r['me']:
        y = obs[r['day']]['last_day'].get('you') or {}
        if 'profit' in y:
            return y['profit']
    rev = r['sold'] * r['price']
    cost = r['meals'] * r['cost_u']
    salv = 0.25 * (r['meals'] - r['sold']) * r['cost_u']
    w = (r['emp'] if r['emp'] is not None else 0) * wage_of(obs, r['day'], r['id'])
    return rev - cost + salv - w - move_prev


def add_profit(rows, obs):
    last = {}
    mc = (obs[max(obs)].get('rules') or {}).get('move_costs', {})
    for r in sorted(rows, key=lambda r: (r['id'], r['day'])):
        prev = last.get(r['id'])
        m = 0.0
        if prev and prev['spot'] != r['spot'] and r['day'] == prev['day'] + 1:
            m = mc.get(prev['spot'], {}).get(r['spot'], 0)
        r['move'] = m
        r['profit'] = est_profit(r, obs, m)
        last[r['id']] = r


def sec1(rows, ev):
    print('## 1. Price response\n')
    me = sorted([r for r in rows if r['me']], key=lambda r: r['day'])
    print('### 1a. Our orders, per day (conv = sold/effective arrivals; sell-thru = sold/meals)\n')
    table(['day', 'wthr', 'q', 'price', 'rep', 'surplus', 'rivals@spot', 'eff arrivals', 'sold', 'meals',
           'conv', 'sell-thru', 'sellout'],
          [[r['day'], r['weather'], r['q'][:3], f(r['price'], 2), f(r['rep'], 0), f(r['surplus'], 2),
            r['n'] - 1, r['eff'], r['sold'], r['meals'], f(r['sold'] / r['eff'], 2) if r['eff'] else '-',
            f(r['sold'] / r['meals'], 2), 'Y' if r['sellout'] else ''] for r in me])
    print('### 1b. Our days by (quality, price bin), non-sellout days only (uncensored conversion)\n')
    bins = defaultdict(list)
    for r in me:
        if r['sellout']:
            continue
        b = '<=8.5' if r['price'] <= 8.5 else '8.5-9.5' if r['price'] <= 9.5 else '9.5-10.5' if r['price'] <= 10.5 else '>10.5'
        bins[(r['q'], b)].append(r)
    table(['quality', 'price', 'n days', 'mean conv', 'mean sell-thru', 'mean surplus'],
          [[k[0], k[1], len(v), f(mean([r['sold'] / r['eff'] for r in v if r['eff']]), 2),
            f(mean([r['sold'] / r['meals'] for r in v]), 2), f(mean([r['surplus'] for r in v]), 2)]
           for k, v in sorted(bins.items())])
    print('### 1c. All trucks, non-sellout, alone at spot, no surprise event at spot (clean demand curve): conversion = sold/eff arrivals by surplus bin\n')
    sb = defaultdict(list)
    for r in rows:
        if r['n'] == 1 and not r['sellout'] and r['eff'] >= 30 and not r['tag']:
            s = r['surplus']
            b = '<4' if s < 4 else '4-5' if s < 5 else '5-6' if s < 6 else '6-7' if s < 7 else '7-8' if s < 8 else '8-9' if s < 9 else '9-10' if s < 10 else '10-11' if s < 11 else '>=11'
            sb[b].append(r)
    order = ['<4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10-11', '>=11']
    table(['surplus (value-price)', 'n truck-days', 'mean conv', 'min', 'max', 'mean price'],
          [[b, len(sb[b]), f(mean([r['sold'] / r['eff'] for r in sb[b]]), 2),
            f(min(r['sold'] / r['eff'] for r in sb[b]), 2), f(max(r['sold'] / r['eff'] for r in sb[b]), 2),
            f(mean([r['price'] for r in sb[b]]), 2)] for b in order if sb[b]])
    print('### 1d. Same, by quality and price (alone, non-sellout)\n')
    qb = defaultdict(list)
    for r in rows:
        if r['n'] == 1 and not r['sellout'] and r['eff'] >= 30 and not r['tag']:
            b = int(r['price'] // 1.5 * 1.5 * 10) / 10.0
            qb[(r['q'], b)].append(r)
    table(['quality', 'price>=', 'n', 'mean conv', 'mean surplus'],
          [[k[0], k[1], len(v), f(mean([r['sold'] / r['eff'] for r in v]), 2), f(mean([r['surplus'] for r in v]), 2)]
           for k, v in sorted(qb.items()) if len(v) >= 2])


def sec2(rows, ev, obs):
    print('## 2. Prediction error structure (evaluation-v2: expected vs actual sales)\n')
    byday = {r['day']: r for r in rows if r['me']}
    rs = []
    for d, e in sorted(ev.items()):
        m = byday.get(d)
        if not m:
            continue
        rs.append(dict(day=d, wthr=m['weather'], q=m['q'], price=m['price'], exp=e['expected_sales'], act=e['actual_sales'],
                       sell=e['sellout'], meals=m['meals'], ratio_c=(m['cust'] / m['fexp']) if m['fexp'] else None,
                       rivals=m['n'] - 1, tag=m['tag'], eprof=e.get('expected_profit'), aprof=e.get('actual_profit'), cust=m['cust'], fexp=m['fexp']))
    table(['day', 'wthr', 'price', 'q', 'exp sales', 'act sales', 'act/exp', 'sellout', 'cust/forecast', 'exp profit', 'act profit', 'surprise'],
          [[r['day'], r['wthr'], f(r['price'], 2), r['q'][:3], f(r['exp']), r['act'], f(r['act'] / r['exp'], 2),
            'Y' if r['sell'] else '', f(r['ratio_c'], 2), f(r['eprof'], 0), f(r['aprof'], 0), r['tag']] for r in rs])

    def grp(label, keyf):
        g = defaultdict(list)
        for r in rs:
            g[keyf(r)].append(r)
        table([label, 'n', 'mean act/exp', 'mean (act-exp)', 'mean profit act-exp'],
              [[k, len(v), f(mean([r['act'] / r['exp'] for r in v]), 2), f(mean([r['act'] - r['exp'] for r in v]), 1),
                f(mean([(r['aprof'] or 0) - (r['eprof'] or 0) for r in v]), 0)] for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))])
    grp('weather', lambda r: r['wthr'])
    grp('sellout (censored)', lambda r: 'sellout' if r['sell'] else 'no sellout')
    grp('price bin', lambda r: '<=8.5' if r['price'] <= 8.5 else '8.5-9.5' if r['price'] <= 9.5 else '>9.5')
    grp('crowd vs forecast', lambda r: 'n/a' if r['ratio_c'] is None else 'short (<0.9)' if r['ratio_c'] < .9 else 'over (>1.1)' if r['ratio_c'] > 1.1 else 'ok')
    print('### 2b. Actual customers / forecast expected_customers (all spots, all days with forecast)\n')
    seen = {}
    for r in rows:
        if r['fexp'] and not r['tag']:
            seen[(r['day'], r['spot'])] = (r['weather'], r['spot'], r['cust'] / r['fexp'], r['cust'], r['fexp'])
    # include empty spots from raw obs
    for d, x in obs.items():
        prev = obs.get(d - 1)
        fc = (prev or {}).get('forecast') or {}
        if fc.get('day') != d:
            continue
        for sp, s in x['last_day']['spots'].items():
            e = fc['expected_customers'].get(sp)
            tg = {m.get('kind') for m in (x['last_day'].get('moments') or []) if isinstance(m, dict) and m.get('spot') == sp}
            if e and not (tg & {'firedrill', 'giveaway', 'protest'}):
                seen[(d, sp)] = (x['last_day'].get('weather'), sp, s['customers'] / e, s['customers'], e)
    for label, ix in (('weather', 0), ('spot', 1)):
        g = defaultdict(list)
        for v in seen.values():
            g[v[ix]].append(v)
        table([label, 'n spot-days', 'mean cust/forecast', 'sd', 'total cust/total forecast'],
              [[k, len(v), f(mean([x[2] for x in v]), 2),
                f(math.sqrt(mean([(x[2] - mean([y[2] for y in v])) ** 2 for x in v])), 2),
                f(sum(x[3] for x in v) / sum(x[4] for x in v), 2)] for k, v in sorted(g.items())])


def sec2c(obs):
    print('### 2c. Forecast weather probabilities vs realized (does the forecast know the weather?)\n')
    out = []
    for d in sorted(obs):
        prev = obs.get(d - 1)
        fc = (prev or {}).get('forecast') or {}
        if fc.get('day') != d:
            continue
        w = fc.get('weather') or {}
        sp = obs[d]['last_day']['spots']
        cu = sum(s['customers'] for s in sp.values())
        fe = sum(fc['expected_customers'].values())
        out.append((d, w.get('clear', 0), w.get('rain', 0), obs[d]['last_day'].get('weather'), cu / fe))
    table(['day', 'P(clear)', 'P(rain)', 'realized', 'total cust / total forecast'],
          [[d, f(a, 2), f(b, 2), c, f(r, 2)] for d, a, b, c, r in out])
    g = defaultdict(list)
    for d, a, b, c, r in out:
        g[c].append(r)
    table(['realized weather', 'days', 'mean total cust/forecast'], [[k, len(v), f(mean(v), 2)] for k, v in sorted(g.items())])


def sec3(rows, obs):
    print('## 3. Rivals\n')
    ids = sorted({r['id'] for r in rows}, key=lambda i: str([r['name'] for r in rows if r['id'] == i][:1]))
    summ = []
    for i in ids:
        rr = sorted([r for r in rows if r['id'] == i], key=lambda r: r['day'])
        nm = rr[0]['name'] or i
        spots = defaultdict(int)
        for r in rr:
            spots[r['spot']] += 1
        qs = defaultdict(int)
        for r in rr:
            qs[r['q'][:3]] += 1
        summ.append([nm, len(rr), ', '.join('%s %d' % (k, v) for k, v in sorted(spots.items(), key=lambda kv: -kv[1])),
                     ', '.join('%s %d' % kv for kv in sorted(qs.items(), key=lambda kv: -kv[1])),
                     f(mean([r['price'] for r in rr]), 2), f(mean([r['meals'] for r in rr]), 0),
                     f(mean([r['sold'] for r in rr]), 0),
                     f(sum(r['sellout'] for r in rr) / len(rr), 2),
                     f(mean([r['profit'] for r in rr]), 0),
                     sum(1 for a, b in zip(rr, rr[1:]) if a['spot'] != b['spot'] and b['day'] == a['day'] + 1)])
    table(['truck', 'days seen', 'spots (days)', 'quality', 'mean price', 'mean meals', 'mean sold', 'sellout rate',
           'est profit/day', 'moves'], summ)
    print('### 3b. Cash growth per day (true profit incl. taxes/wages/everything), first to last observation\n')
    d0, d1 = min(obs), max(obs)
    c = {}
    for d in (d0, d1):
        for t in obs[d].get('trucks', []):
            if isinstance(t, dict) and 'cash' in t:
                c.setdefault(t['id'], {})[d] = t['cash']
        if (obs[d].get('you') or {}).get('cash') is not None:
            c.setdefault(ME, {})[d] = obs[d]['you']['cash']
    nm = {r['id']: r['name'] for r in rows}
    out = []
    # use latest day with cash for all
    dd = [d for d in sorted(obs) if all(isinstance(t, dict) and 'cash' in t for t in obs[d].get('trucks', []))]
    for a in [x for x in (dd[0], 15, 20) if x in dd]:
        b = dd[-1]
        out = []
        for t in obs[b]['trucks']:
            ca = next((x['cash'] for x in obs[a]['trucks'] if x['id'] == t['id']), None)
            if ca is not None:
                out.append([nm.get(t['id'], t['id']), a, b, f(ca, 0), f(t['cash'], 0), f((t['cash'] - ca) / (b - a), 1)])
        table(['truck', 'from day', 'to day', 'cash from', 'cash to', 'cash gain/day'], sorted(out, key=lambda r: -float(r[5])))
    print('### 3c. Rival arrivals at the spot we occupy (we sit at Thompson almost always)\n')
    me = {r['day']: r for r in rows if r['me']}
    eps = []
    for r in sorted(rows, key=lambda r: r['day']):
        if r['me'] or r['day'] not in me:
            continue
        m = me[r['day']]
        if r['spot'] != m['spot']:
            continue
        prev = [x for x in rows if x['id'] == r['id'] and x['day'] == r['day'] - 1]
        came_from = prev[0]['spot'] if prev else '?'
        eps.append([r['day'], r['name'], came_from, r['q'][:3], f(r['price'], 2), r['meals'], r['sold'],
                    '%s %s %s' % (m['q'][:3], f(m['price'], 2), m['sold']), f(m['rep'], 0), f(r['rep'], 0)])
    table(['day', 'rival', 'came from', 'q', 'price', 'meals', 'sold', 'us (q price sold)', 'our rep', 'its rep'], eps)
    print('### 3e. Rival moves onto a spot we occupied the day before (we left it) or onto our spot the same day\n')
    byid = defaultdict(dict)
    for r in rows:
        byid[r['id']][r['day']] = r
    mine = byid[ME]
    ep = []
    for r in rows:
        if r['me']:
            continue
        pr = byid[r['id']].get(r['day'] - 1)
        m0 = mine.get(r['day'] - 1)
        if pr and m0 and pr['spot'] != r['spot'] and m0['spot'] == r['spot']:
            m1 = mine.get(r['day'])
            ep.append([r['day'], r['name'], r['spot'], 'we stayed' if m1 and m1['spot'] == r['spot'] else 'we left', r['q'][:3], f(r['price'], 2), r['sold'], r['meals'],
                       ('%s %s %s/%s' % (m1['q'][:3], f(m1['price'], 2), m1['sold'], m1['meals'])) if m1 else '-'])
    table(['day', 'rival', 'spot', 'us', 'q', 'price', 'sold', 'meals', 'our result (q price sold/meals)'], ep)
    print('### 3d. Rival sellouts: mean price and spot (sellout rows only)\n')
    g = defaultdict(list)
    for r in rows:
        if not r['me']:
            g[(r['name'], 'sellout' if r['sellout'] else 'not')].append(r)
    table(['rival', 'outcome', 'n', 'mean price', 'mean meals', 'mean eff arrivals', 'mean n at spot'],
          [[k[0], k[1], len(v), f(mean([r['price'] for r in v]), 2), f(mean([r['meals'] for r in v]), 0),
            f(mean([r['eff'] for r in v]), 0), f(mean([r['n'] for r in v]), 1)] for k, v in sorted(g.items())])


def sec4(rows, obs):
    print('## 4. Profit decomposition, our days (from last_day.you)\n')
    out = []
    for d in sorted(obs):
        y = obs[d]['last_day'].get('you') or {}
        if y.get('status') != 'played' or 'profit' not in y:
            continue
        out.append((d, y))
    keys = ['revenue', 'cost', 'salvage', 'wages', 'move_cost', 'taxes', 'subsidies', 'fines']
    def row(d, y):
        return [d, y['spot'], y['quality'][:3], f(y['price'], 2), '%s/%s' % (y['sold'], y['meals'])] + \
               [f(y.get(k, 0) or 0, 0) for k in keys] + [f(y['profit'], 0)]
    srt = sorted(out, key=lambda t: t[1]['profit'])
    head = ['day', 'spot', 'q', 'price', 'sold/meals'] + keys + ['profit']
    print('Worst 6 days:\n')
    table(head, [row(d, y) for d, y in srt[:6]])
    print('Best 6 days:\n')
    table(head, [row(d, y) for d, y in srt[-6:][::-1]])
    played = [y for d, y in out if d >= 15]
    print('Mean over days >=15 (n=%d): ' % len(played) + ', '.join('%s %s' % (k, f(mean([y.get(k, 0) or 0 for y in played]), 1)) for k in keys + ['profit']) + '\n')
    print('Profit by profile (days >= 15):\n')
    g = defaultdict(list)
    for d, y in out:
        if d >= 15:
            g[(y['quality'], 'sellout' if y['sold'] >= y['meals'] else 'unsold')].append(y['profit'])
    table(['quality', 'outcome', 'n', 'mean profit'], [[k[0], k[1], len(v), f(mean(v), 0)] for k, v in sorted(g.items())])


def sec5(rows):
    print('## 5. Competition effect (all trucks, all days)\n')
    g = defaultdict(list)
    for r in rows:
        g[r['n']].append(r)
    table(['trucks at spot', 'truck-days', 'mean eff arrivals', 'mean sold/truck', 'mean est profit/truck', 'sellout rate', 'mean price', 'mean surplus'],
          [[k, len(v), f(mean([r['eff'] for r in v]), 0), f(mean([r['sold'] for r in v]), 1), f(mean([r['profit'] for r in v]), 0),
            f(sum(r['sellout'] for r in v) / len(v), 2), f(mean([r['price'] for r in v]), 2), f(mean([r['surplus'] for r in v]), 2)]
           for k, v in sorted(g.items())])
    print('Share of spot arrivals served and per-truck share (spot-day level, n>=1):\n')
    sd = {}
    for r in rows:
        sd[(r['day'], r['spot'])] = r
    g = defaultdict(list)
    for r in sd.values():
        if r['eff'] > 0:
            g[r['n']].append(r['tot_sold'] / r['eff'])
    table(['trucks at spot', 'spot-days', 'mean served / eff arrivals'], [[k, len(v), f(mean(v), 2)] for k, v in sorted(g.items())])
    print('Same, controlling for spot (mean sold per truck by spot and n):\n')
    g = defaultdict(list)
    for r in rows:
        g[(r['spot'], min(r['n'], 3))].append(r)
    table(['spot', 'n (3=3+)', 'truck-days', 'mean eff arrivals', 'mean sold', 'mean est profit'],
          [[k[0], k[1], len(v), f(mean([r['eff'] for r in v]), 0), f(mean([r['sold'] for r in v]), 1), f(mean([r['profit'] for r in v]), 0)]
           for k, v in sorted(g.items())])


def sec6(rows, obs):
    print('## 6. Walk-over (days >= 16 only: fields absent before)\n')
    sd = {}
    for r in rows:
        sd[(r['day'], r['spot'])] = r
    spots = defaultdict(lambda: dict(wi=0, wo=0, sold=0, sdays=0, wdays=0, cust=0, eday=0))
    for d, x in obs.items():
        if d < 16:  # walk fields only exist from day 16
            continue
        for sp, s in x['last_day']['spots'].items():
            a = spots[sp]
            a['wi'] += s.get('walked_in', 0) or 0
            a['wo'] += s.get('walked_out', 0) or 0
            a['cust'] += s['customers']
            a['eday'] += 1
            ts = sum(t['sold'] for t in s['trucks'])
            if s['trucks']:
                a['sold'] += ts
                a['sdays'] += 1
                a['wi_t'] = a.get('wi_t', 0) + (s.get('walked_in', 0) or 0)
            else:
                a['wo_empty'] = a.get('wo_empty', 0) + (s.get('walked_out', 0) or 0)
    table(['spot', 'days', 'days with trucks', 'cust', 'walked_out/day', 'walked_in/day', 'walk-in share of sales (days w/ trucks)', 'walked_out from empty spot-days'],
          [[sp, a['eday'], a['sdays'], a['cust'], f(a['wo'] / a['eday']), f(a['wi'] / a['eday']),
            f(a.get('wi_t', 0) / a['sold'], 2) if a['sold'] else '-', a.get('wo_empty', 0)] for sp, a in sorted(spots.items())])
    tot_wi = sum(a.get('wi_t', 0) for a in spots.values())
    tot_sold = sum(a['sold'] for a in spots.values())
    print('Overall: walked-in / total sold = %s/%s = %s\n' % (tot_wi, tot_sold, f(tot_wi / tot_sold, 3)))
    print('Our truck: sales share that are walk-ins (walked_in / our-spot effective arrivals) by n trucks at spot\n')
    g = defaultdict(list)
    for r in rows:
        if r['me'] and r['day'] >= 16:
            g[r['n']].append(r)
    table(['trucks at our spot', 'days', 'mean walked_in', 'mean walked_out', 'mean our sold', 'mean walked_in/total sold at spot'],
          [[k, len(v), f(mean([r['wi'] for r in v])), f(mean([r['wo'] for r in v])), f(mean([r['sold'] for r in v])),
            f(sum(r['wi'] for r in v) / max(1, sum(r['tot_sold'] for r in v)), 2)] for k, v in sorted(g.items())])


def sec7(rows):
    print('## 7. Best rival strategies by est profit/day (days >= 15, all trucks incl. ours; crew/wage data from day 22)\n')
    rr = [r for r in rows if r['day'] >= 15]
    g = defaultdict(list)
    for r in rr:
        g[(r['name'], r['spot'], r['q'][:3])].append(r)
    table(['truck', 'spot', 'q', 'truck-days', 'mean price', 'mean sold', 'sellout', 'mean est profit'],
          sorted([[k[0], k[1], k[2], len(v), f(mean([r['price'] for r in v]), 2), f(mean([r['sold'] for r in v]), 0),
                   f(sum(r['sellout'] for r in v) / len(v), 2), f(mean([r['profit'] for r in v]), 0)] for k, v in g.items() if len(v) >= 3],
                 key=lambda r: -float(r[-1])))
    print('Profit by price bin and quality (all trucks, days>=15, n>=5):\n')
    g = defaultdict(list)
    for r in rr:
        g[(r['q'][:3], int(r['price']))].append(r)
    table(['q', 'price floor', 'n', 'mean sold', 'sellout', 'mean est profit'],
          [[k[0], k[1], len(v), f(mean([r['sold'] for r in v]), 0), f(sum(r['sellout'] for r in v) / len(v), 2), f(mean([r['profit'] for r in v]), 0)]
           for k, v in sorted(g.items()) if len(v) >= 5])
    print('Profit by crew size (all trucks, days>=15):\n')
    g = defaultdict(list)
    for r in rr:
        g[r['emp']].append(r)
    table(['employees', 'n', 'mean meals', 'mean sold', 'mean est profit'],
          [[k, len(v), f(mean([r['meals'] for r in v]), 0), f(mean([r['sold'] for r in v]), 0), f(mean([r['profit'] for r in v]), 0)]
           for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--section', type=int, default=0)
    a = ap.parse_args()
    obs, ev = load(a.data_root)
    if not obs:
        raise SystemExit('no observations under %s' % a.data_root)
    rows = build(obs)
    latest = {}
    for d in sorted(obs):
        for t in obs[d].get('trucks', []):
            if isinstance(t, dict) and t.get('truck_name'):
                latest[t['id']] = t['truck_name']
    for r in rows:
        r['name'] = latest.get(r['id'], r['name'] or r['id'])
    add_profit(rows, obs)
    print('# Live analysis: %d observed days (%d..%d), %d truck-days\n' % (len(obs), min(obs), max(obs), len(rows)))
    secs = {1: lambda: sec1(rows, ev), 2: lambda: (sec2(rows, ev, obs), sec2c(obs)), 3: lambda: sec3(rows, obs),
            4: lambda: sec4(rows, obs), 5: lambda: sec5(rows), 6: lambda: sec6(rows, obs), 7: lambda: sec7(rows)}
    for k, fn in secs.items():
        if a.section in (0, k):
            fn()


if __name__ == '__main__':
    main()
