"""
 SYNTHETIC E-COMMERCE CUSTOMER BEHAVIOUR DATASET GENERATOR
 Npontu Technologies - Intelligent Systems Services Engineer assignment

 Simulates 18 months (2025-01-01 -> 2026-06-30) of activity on a Ghana-based
 e-commerce platform, across six related tables:

     customers.csv     demographics + acquisition
     products.csv      catalogue
     events.csv        clickstream / browsing behaviour   (the "big data" table)
     orders.csv        order headers
     order_items.csv   order line items
     reviews.csv       post-purchase product interactions

 The data is deliberately DIRTY. Missing values, duplicate rows, near-duplicate
 customers, mixed date formats, currency strings, sentinel values (-999, "N/A"),
 inconsistent categorical spellings, impossible ages, negative quantities and
 referential violations are all injected on purpose so that the cleaning and
 feature-engineering stages have something real to do. Every injected defect is
 counted and reported at the end of this script.

 There is genuine latent signal underneath the noise: each customer carries a
 hidden `engagement` score and a hidden churn date, both driven by acquisition
 channel, loyalty tier, discount dependence and delivery experience. A churn
 model built on well-engineered RFM + funnel features should reach a clearly
 better-than-random AUC. A model built on raw, uncleaned columns should not.

 Reproducible: fixed seed (42). Re-running produces byte-identical files.
"""

import json
import os
import numpy as np
import pandas as pd

# CONFIG 
SEED = 42
rng = np.random.default_rng(SEED)

OUT_DIR = "/mnt/user-data/outputs/ecommerce_dataset"
os.makedirs(OUT_DIR, exist_ok=True)

N_CUSTOMERS = 20_000
N_PRODUCTS = 1_500
N_BROWSE_EVENTS = 460_000        # purchase/checkout events are added on top

WINDOW_START = pd.Timestamp("2025-01-01")
WINDOW_END = pd.Timestamp("2026-06-30 23:59:59")
SNAPSHOT = WINDOW_END            # "today" from the dataset's point of view

# Defect counters -> printed as a verification log at the end
defects = {}


def log(name, n):
    """Record how many defects of a given kind were injected."""
    defects[name] = defects.get(name, 0) + int(n)


def pick_idx(n_rows, frac, rng):
    """Random row positions covering `frac` of n_rows (no repeats)."""
    k = int(round(n_rows * frac))
    if k <= 0:
        return np.array([], dtype=int)
    return rng.choice(n_rows, size=k, replace=False)


def sprinkle_nulls(df, col, frac, rng, label=None):
    """Blank out a fraction of a column with real NaN."""
    idx = pick_idx(len(df), frac, rng)
    df.loc[df.index[idx], col] = np.nan
    log(label or f"nulls::{col}", len(idx))


def sprinkle_values(df, col, frac, rng, values, label):
    """Overwrite a fraction of a column with junk sentinel values."""
    idx = pick_idx(len(df), frac, rng)
    df[col] = df[col].astype(object)
    df.loc[df.index[idx], col] = rng.choice(values, size=len(idx))
    log(label, len(idx))


def messy_dates(series, frac_slash, frac_long, rng, label):
    """Rewrite some ISO date strings into other formats. Returns object series."""
    s = series.astype(object).copy()
    valid = s.notna().to_numpy().nonzero()[0]
    n_slash = int(len(valid) * frac_slash)
    n_long = int(len(valid) * frac_long)
    chosen = rng.choice(valid, size=n_slash + n_long, replace=False)
    slash_pos, long_pos = chosen[:n_slash], chosen[n_slash:]

    dt = pd.to_datetime(s, errors="coerce", format="mixed")
    s.iloc[slash_pos] = dt.iloc[slash_pos].dt.strftime("%d/%m/%Y")
    s.iloc[long_pos] = dt.iloc[long_pos].dt.strftime("%B %d, %Y")
    log(label, len(chosen))
    return s


def money_strings(series, frac, rng, label, prefix="GHS "):
    """Turn some numeric money values into 'GHS 1,234.50' strings."""
    s = series.astype(object).copy()
    valid = pd.to_numeric(s, errors="coerce").notna().to_numpy().nonzero()[0]
    chosen = rng.choice(valid, size=int(len(valid) * frac), replace=False)
    s.iloc[chosen] = [f"{prefix}{float(v):,.2f}" for v in s.iloc[chosen]]
    log(label, len(chosen))
    return s


def duplicate_rows(df, frac, rng, label, shuffle=True):
    """Append exact duplicate rows, then optionally reshuffle so they are not adjacent."""
    idx = pick_idx(len(df), frac, rng)
    dupes = df.iloc[idx].copy()
    out = pd.concat([df, dupes], ignore_index=True)
    log(label, len(idx))
    if shuffle:
        out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return out


#  GHANA REFERENCE DATA 
CITIES = {
    "Accra": "Greater Accra", "Tema": "Greater Accra", "Madina": "Greater Accra",
    "Kumasi": "Ashanti", "Obuasi": "Ashanti", "Takoradi": "Western",
    "Cape Coast": "Central", "Tamale": "Northern", "Ho": "Volta",
    "Sunyani": "Bono", "Koforidua": "Eastern", "Bolgatanga": "Upper East",
    "Wa": "Upper West", "Techiman": "Bono East",
}
CITY_NAMES = list(CITIES)
CITY_WEIGHTS = np.array([28, 7, 5, 16, 3, 8, 5, 7, 4, 4, 5, 3, 2, 3], float)
CITY_WEIGHTS /= CITY_WEIGHTS.sum()

PAYMENTS = ["MTN Mobile Money", "Telecel Cash", "AirtelTigo Money",
            "Debit Card", "Bank Transfer", "Cash on Delivery"]
PAYMENT_W = np.array([0.42, 0.14, 0.08, 0.16, 0.06, 0.14])

CHANNELS = ["Organic Search", "Paid Social", "Referral", "Email Campaign",
            "Influencer", "Direct", "Marketplace"]
CHANNEL_W = np.array([0.24, 0.22, 0.12, 0.10, 0.11, 0.13, 0.08])
# Hidden channel quality -> drives engagement and churn. Paid/influencer traffic
# converts worse and churns faster; referral and email are the strongest.
CHANNEL_QUALITY = {"Organic Search": 1.05, "Paid Social": 0.72, "Referral": 1.35,
                   "Email Campaign": 1.20, "Influencer": 0.78, "Direct": 1.10,
                   "Marketplace": 0.85}

DEVICES = ["Android", "iOS", "Web Mobile", "Web Desktop"]
DEVICE_W = np.array([0.55, 0.13, 0.19, 0.13])

TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
TIER_W = np.array([0.52, 0.28, 0.14, 0.06])
TIER_BOOST = {"Bronze": 0.85, "Silver": 1.0, "Gold": 1.25, "Platinum": 1.55}

CATEGORIES = {
    "Phones & Tablets": (["Smartphones", "Tablets", "Phone Accessories", "Power Banks"], 180, 6500),
    "Computing":        (["Laptops", "Monitors", "Printers", "Storage"], 300, 12000),
    "Electronics":      (["Televisions", "Audio", "Cameras", "Small Appliances"], 120, 9000),
    "Fashion":          (["Men's Clothing", "Women's Clothing", "Footwear", "Bags"], 35, 900),
    "Home & Kitchen":   (["Cookware", "Furniture", "Bedding", "Cleaning"], 25, 3500),
    "Health & Beauty":  (["Skincare", "Haircare", "Fragrance", "Supplements"], 15, 600),
    "Groceries":        (["Beverages", "Pantry", "Snacks", "Household"], 8, 250),
    "Baby Products":    (["Diapers", "Baby Food", "Toys", "Baby Care"], 20, 800),
    "Sports & Fitness": (["Gym Equipment", "Sportswear", "Outdoor", "Supplements"], 30, 2500),
    "Books & Stationery": (["Textbooks", "Fiction", "Office Supplies", "Art"], 10, 400),
}
BRANDS = ["Samsung", "Tecno", "Infinix", "itel", "Nasco", "Midea", "Hisense",
          "HP", "Dell", "Lenovo", "Apple", "Nivea", "Unilever", "Nestle",
          "Kasapreko", "Fan Milk", "Woodin", "GTP", "Puma", "Adidas", "Generic"]

FIRST_NAMES = ["Kwame", "Ama", "Kofi", "Akosua", "Yaw", "Abena", "Kojo", "Adwoa",
               "Kwabena", "Afia", "Kwaku", "Esi", "Fiifi", "Adjoa", "Nana",
               "Emmanuel", "Grace", "Daniel", "Comfort", "Michael", "Mary",
               "Samuel", "Elizabeth", "Joseph", "Patience", "Isaac", "Gifty",
               "Ibrahim", "Fatima", "Abdul", "Zainab", "Selorm", "Dzifa",
               "Arinze", "Chidi", "Ngozi", "Sena", "Yayra", "Kelvin", "Priscilla"]
LAST_NAMES = ["Mensah", "Osei", "Boateng", "Owusu", "Asante", "Appiah", "Annan",
              "Darko", "Frimpong", "Agyeman", "Adjei", "Ofori", "Amoah", "Baidoo",
              "Quartey", "Tetteh", "Lartey", "Nkrumah", "Danso", "Yeboah",
              "Abubakar", "Alhassan", "Sulemana", "Dogbe", "Ahiable", "Nyarko"]


# 1. PRODUCTS

print("Generating products ...")

cat_names = list(CATEGORIES)
cat_pick = rng.choice(len(cat_names), size=N_PRODUCTS,
                      p=np.array([.12, .08, .11, .17, .12, .11, .10, .06, .07, .06]))

prod_rows = []
for i in range(N_PRODUCTS):
    cat = cat_names[cat_pick[i]]
    subs, lo, hi = CATEGORIES[cat]
    sub = subs[rng.integers(len(subs))]
    # log-uniform price so each category has a realistic long right tail
    price = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    price = round(price, 2)
    margin = rng.uniform(0.12, 0.45)
    prod_rows.append({
        "product_id": f"PRD{i + 1:05d}",
        "product_name": f"{BRANDS[rng.integers(len(BRANDS))]} {sub[:-1] if sub.endswith('s') else sub} {rng.integers(100, 999)}",
        "category": cat,
        "subcategory": sub,
        "brand": BRANDS[rng.integers(len(BRANDS))],
        "unit_price_ghs": price,
        "cost_price_ghs": round(price * (1 - margin), 2),
        "launch_date": (WINDOW_START - pd.Timedelta(days=int(rng.integers(0, 1100)))).date().isoformat(),
        "stock_quantity": int(rng.integers(0, 900)),
        "avg_rating": round(float(np.clip(rng.normal(4.05, 0.55), 1, 5)), 2),
        "is_active": bool(rng.random() > 0.08),
    })

products = pd.DataFrame(prod_rows)
# hidden per-product popularity -> drives view and purchase weights
prod_pop = rng.gamma(1.4, 1.0, N_PRODUCTS)
prod_pop /= prod_pop.sum()
price_lookup = dict(zip(products.product_id, products.unit_price_ghs))


# 2. CUSTOMERS  (+ hidden latent variables that drive everything downstream)

print("Generating customers ...")

cust_ids = np.array([f"CUST{i + 1:06d}" for i in range(N_CUSTOMERS)])

# Signup dates: platform is growing, so more recent months acquire more users.
month_starts = pd.date_range(WINDOW_START, WINDOW_END, freq="MS")
growth = np.linspace(1.0, 2.1, len(month_starts))
# 35% of the base existed before the observation window opened
pre_existing = rng.random(N_CUSTOMERS) < 0.35
signup = np.empty(N_CUSTOMERS, dtype="datetime64[s]")
n_pre = int(pre_existing.sum())
signup[pre_existing] = (WINDOW_START - pd.to_timedelta(
    rng.integers(1, 900, n_pre), unit="D")).values.astype("datetime64[s]")
m_pick = rng.choice(len(month_starts), size=N_CUSTOMERS - n_pre, p=growth / growth.sum())
signup[~pre_existing] = (month_starts[m_pick] + pd.to_timedelta(
    rng.integers(0, 28, N_CUSTOMERS - n_pre), unit="D")).values.astype("datetime64[s]")
signup = pd.to_datetime(signup)

channel = rng.choice(CHANNELS, size=N_CUSTOMERS, p=CHANNEL_W)
tier = rng.choice(TIERS, size=N_CUSTOMERS, p=TIER_W)
device = rng.choice(DEVICES, size=N_CUSTOMERS, p=DEVICE_W)
city = rng.choice(CITY_NAMES, size=N_CUSTOMERS, p=CITY_WEIGHTS)

# latent engagement: the single most important hidden driver 
q = np.array([CHANNEL_QUALITY[c] for c in channel])
t = np.array([TIER_BOOST[x] for x in tier])
engagement = np.clip(rng.beta(2.0, 3.2, N_CUSTOMERS) * q * t, 0.02, 1.0)

# discount dependence: heavy discount users churn more once promos stop
discount_affinity = np.clip(rng.beta(2, 5, N_CUSTOMERS) + (1 - engagement) * 0.25, 0, 1)

age = np.clip(rng.normal(31, 9.5), 16, 78, out=None).astype(int) if False else \
    np.clip(rng.normal(31, 9.5, N_CUSTOMERS), 16, 78).astype(int)
dob = pd.to_datetime(SNAPSHOT) - pd.to_timedelta(
    age * 365 + rng.integers(0, 364, N_CUSTOMERS), unit="D")

income = np.clip(rng.lognormal(9.6, 0.55, N_CUSTOMERS), 1200, 400_000).round(2)

customers = pd.DataFrame({
    "customer_id": cust_ids,
    "full_name": [f"{FIRST_NAMES[i]} {LAST_NAMES[j]}" for i, j in
                  zip(rng.integers(0, len(FIRST_NAMES), N_CUSTOMERS),
                      rng.integers(0, len(LAST_NAMES), N_CUSTOMERS))],
    "gender": rng.choice(["Male", "Female", "Other"], N_CUSTOMERS, p=[.49, .49, .02]),
    "date_of_birth": dob.date.astype(str),
    "city": city,
    "region": [CITIES[c] for c in city],
    "signup_date": signup.date.astype(str),
    "acquisition_channel": channel,
    "preferred_device": device,
    "email_subscribed": rng.random(N_CUSTOMERS) < 0.63,
    "loyalty_tier": tier,
    "annual_income_ghs": income,
    "household_size": np.clip(rng.poisson(3.6, N_CUSTOMERS), 1, 12),
})
customers["email"] = [
    f"{n.split()[0].lower()}.{n.split()[1].lower()}{rng.integers(1, 999)}@{d}"
    for n, d in zip(customers.full_name,
                    rng.choice(["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"],
                               N_CUSTOMERS, p=[.68, .16, .11, .05]))
]
customers = customers[["customer_id", "full_name", "email", "gender", "date_of_birth",
                       "city", "region", "signup_date", "acquisition_channel",
                       "preferred_device", "email_subscribed", "loyalty_tier",
                       "annual_income_ghs", "household_size"]]

# hidden churn clock 
# Low engagement, poor acquisition channel and heavy discount reliance all raise
# the odds of going quiet. `active_end` is when a customer stops doing anything.
churn_risk = np.clip(0.62 - 0.55 * engagement + 0.22 * discount_affinity
                     + rng.normal(0, 0.09, N_CUSTOMERS), 0.02, 0.95)
will_churn = rng.random(N_CUSTOMERS) < churn_risk

tenure_days = np.maximum((SNAPSHOT - signup).days.values, 1)
frac_active = rng.beta(2.2, 1.6, N_CUSTOMERS)          # where in tenure they quit
active_end = signup + pd.to_timedelta(
    np.where(will_churn, (tenure_days * frac_active).astype(int), tenure_days), unit="D")
active_end = pd.Series(active_end).clip(upper=WINDOW_END)
active_start = pd.Series(signup).clip(lower=WINDOW_START)



# 3. SEASONALITY ENGINE

# Hourly demand weights combining: platform growth trend, day-of-week effect,
# hour-of-day effect, and Ghanaian retail calendar spikes.

hours = pd.date_range(WINDOW_START, WINDOW_END, freq="h")
h_idx = np.arange(len(hours))

trend = np.linspace(1.0, 1.95, len(hours))                       # platform growth
dow_f = np.array([0.86, 0.86, 0.91, 0.96, 1.16, 1.32, 1.14])     # Mon..Sun
hour_f = np.array([.25, .16, .11, .09, .10, .18, .38, .62, .82, .92, 1.00, 1.08,
                   1.22, 1.18, 1.02, .96, 1.00, 1.14, 1.34, 1.46, 1.42, 1.18, .78, .44])

w = trend * dow_f[hours.dayofweek.values] * hour_f[hours.hour.values]

# Retail calendar: Black Friday week, Christmas run-up, Independence Day (6 Mar),
# Easter weekend, back-to-school (September), mid-year sales (July).
md = np.stack([hours.month.values, hours.day.values])
w *= np.where((md[0] == 11) & (md[1] >= 24) & (md[1] <= 30), 2.70, 1.0)
w *= np.where((md[0] == 12) & (md[1] <= 24), 1.75, 1.0)
w *= np.where((md[0] == 12) & (md[1] >= 26), 0.65, 1.0)   # post-Christmas lull
w *= np.where((md[0] == 3) & (np.abs(md[1] - 6) <= 2), 1.30, 1.0)
w *= np.where((md[0] == 4) & (md[1] >= 17) & (md[1] <= 21), 1.35, 1.0)
w *= np.where(md[0] == 9, 1.18, 1.0)
w *= np.where((md[0] == 7) & (md[1] <= 7), 1.25, 1.0)
w *= np.where(md[0] == 1, 0.82, 1.0)                     # January squeeze
w_p = w / w.sum()

HSEC = hours.values.astype("datetime64[s]").astype("int64")


def sample_times(starts, ends, rng, max_rounds=16):
    """
    Draw timestamps that follow the seasonality curve AND fall inside each
    row's [start, end] activity window. Rejection sampling: propose from the
    global seasonal distribution, keep the proposals that land in-window,
    re-propose the rest. Stragglers fall back to uniform-in-window.
    """
    s = np.asarray(starts).astype("datetime64[s]").astype("int64")
    e = np.asarray(ends).astype("datetime64[s]").astype("int64")
    e = np.maximum(e, s + 3600)
    n = len(s)
    out = np.zeros(n, dtype="int64")
    todo = np.arange(n)
    for _ in range(max_rounds):
        if todo.size == 0:
            break
        pick = rng.choice(len(HSEC), size=todo.size, p=w_p)
        cand = HSEC[pick] + rng.integers(0, 3600, todo.size)
        ok = (cand >= s[todo]) & (cand <= e[todo])
        out[todo[ok]] = cand[ok]
        todo = todo[~ok]
    if todo.size:
        span = np.maximum(e[todo] - s[todo], 1)
        out[todo] = s[todo] + (rng.random(todo.size) * span).astype("int64")
    return pd.to_datetime(out, unit="s")



# 4. ORDERS  +  ORDER ITEMS

print("Generating orders and order items ...")

active_days = np.maximum((active_end - active_start).dt.days.values, 1)

# Expected purchase count scales with hidden engagement and time spent active.
lam = (0.55 + 10.5 * engagement ** 1.35) * (active_days / 365.0)
lam *= 62_000 / lam.sum()                       # calibrate to a target volume
n_orders_per_cust = rng.poisson(lam)
n_orders_per_cust[(engagement > 0.75) & (n_orders_per_cust == 0)] = 1

order_cust_pos = np.repeat(np.arange(N_CUSTOMERS), n_orders_per_cust)
N_ORDERS = len(order_cust_pos)

order_ts = sample_times(active_start.values[order_cust_pos],
                        active_end.values[order_cust_pos], rng)
order_ids = np.array([f"ORD{i + 1:07d}" for i in range(N_ORDERS)])

# Shipping cost and lead time depend on how far the city is from Accra/Tema hubs.
FAR = {"Tamale": 3, "Bolgatanga": 4, "Wa": 4, "Ho": 2, "Sunyani": 3, "Techiman": 3,
       "Takoradi": 2, "Cape Coast": 2, "Koforidua": 1, "Kumasi": 1, "Obuasi": 2,
       "Accra": 0, "Tema": 0, "Madina": 0}
o_city = customers.city.values[order_cust_pos]
far = np.array([FAR[c] for c in o_city])
ship_fee = np.round(12 + far * 9 + rng.normal(0, 3, N_ORDERS), 2).clip(0, None)
delivery_days = np.clip(np.round(1.6 + far * 1.5 + rng.exponential(1.5, N_ORDERS)), 1, 30)

o_status = rng.choice(["delivered", "shipped", "pending", "cancelled", "returned"],
                      N_ORDERS, p=[0.795, 0.055, 0.030, 0.075, 0.045])
# Customers on their way out have visibly worse order outcomes.
churner_order = will_churn[order_cust_pos]
bump = rng.random(N_ORDERS) < 0.10
o_status = np.where(churner_order & bump,
                    rng.choice(["cancelled", "returned"], N_ORDERS, p=[.55, .45]), o_status)

o_pay = rng.choice(PAYMENTS, N_ORDERS, p=PAYMENT_W)

# line items 
items_per_order = np.clip(1 + rng.poisson(0.85, N_ORDERS), 1, 8)
item_order_pos = np.repeat(np.arange(N_ORDERS), items_per_order)
N_ITEMS = len(item_order_pos)

# Each customer has a favourite category; 55% of lines come from it. This gives
# the recommendation / market-basket angle something real to find.
fav_cat = rng.choice(cat_names, N_CUSTOMERS, p=np.array([.12, .08, .11, .17, .12, .11, .10, .06, .07, .06]))
cat_to_pos = {c: np.where(products.category.values == c)[0] for c in cat_names}
cat_to_p = {c: prod_pop[pos] / prod_pop[pos].sum() for c, pos in cat_to_pos.items()}

item_cust_pos = order_cust_pos[item_order_pos]
use_fav = rng.random(N_ITEMS) < 0.55
prod_pos = rng.choice(N_PRODUCTS, size=N_ITEMS, p=prod_pop)
for c in cat_names:                                   # overwrite the favourite-category lines
    m = use_fav & (fav_cat[item_cust_pos] == c)
    if m.sum():
        prod_pos[m] = rng.choice(cat_to_pos[c], size=int(m.sum()), p=cat_to_p[c])

item_prod_id = products.product_id.values[prod_pos]
item_unit_price = products.unit_price_ghs.values[prod_pos]
# cheap goods get bought in bulk, expensive goods one at a time
qty = np.where(item_unit_price < 100, 1 + rng.poisson(1.7, N_ITEMS),
               np.where(item_unit_price < 800, 1 + rng.poisson(0.45, N_ITEMS), 1)).clip(1, 12)

disc_aff_item = discount_affinity[item_cust_pos]
has_disc = rng.random(N_ITEMS) < (0.18 + 0.45 * disc_aff_item)
disc_pct = np.where(has_disc, np.round(rng.choice([5, 10, 15, 20, 25, 30, 40], N_ITEMS,
                                                  p=[.22, .26, .18, .14, .1, .07, .03]), 2), 0.0)

order_items = pd.DataFrame({
    "order_item_id": [f"ITM{i + 1:08d}" for i in range(N_ITEMS)],
    "order_id": order_ids[item_order_pos],
    "product_id": item_prod_id,
    "quantity": qty,
    "unit_price_ghs": np.round(item_unit_price, 2),
    "discount_pct": disc_pct,
})
order_items["line_total_ghs"] = np.round(
    order_items.quantity * order_items.unit_price_ghs * (1 - order_items.discount_pct / 100), 2)

line_sum = order_items.groupby("order_id", sort=False).line_total_ghs.sum()
line_sum = line_sum.reindex(order_ids).fillna(0).values
disc_amount = np.round(
    order_items.assign(d=lambda d: d.quantity * d.unit_price_ghs * d.discount_pct / 100)
    .groupby("order_id", sort=False).d.sum().reindex(order_ids).fillna(0).values, 2)

promo_codes = ["NEW10", "MOMO5", "FREESHIP", "BF25", "XMAS20", "LOYAL15", "WELCOME200"]
code = np.where(disc_amount > 0,
                rng.choice(promo_codes, N_ORDERS, p=[.2, .16, .14, .16, .14, .12, .08]), None)

orders = pd.DataFrame({
    "order_id": order_ids,
    "customer_id": customers.customer_id.values[order_cust_pos],
    "order_timestamp": order_ts.strftime("%Y-%m-%d %H:%M:%S"),
    "order_status": o_status,
    "payment_method": o_pay,
    "shipping_city": o_city,
    "shipping_fee_ghs": ship_fee,
    "discount_code": code,
    "discount_amount_ghs": disc_amount,
    "items_subtotal_ghs": np.round(line_sum, 2),
    "total_amount_ghs": np.round(line_sum + ship_fee, 2),
    "delivery_days": np.where(np.isin(o_status, ["delivered", "returned"]), delivery_days, np.nan),
})


# 5. CLICKSTREAM EVENTS  (the volume table)

print("Generating clickstream events ...")

ev_lam = (2.0 + 40 * engagement ** 1.15) * (active_days / 365.0)
ev_lam *= N_BROWSE_EVENTS / ev_lam.sum()
n_ev_per_cust = rng.poisson(ev_lam)
ev_cust_pos = np.repeat(np.arange(N_CUSTOMERS), n_ev_per_cust)
N_BROWSE = len(ev_cust_pos)

ev_ts = sample_times(active_start.values[ev_cust_pos],
                     active_end.values[ev_cust_pos], rng)

BROWSE_TYPES = ["page_view", "product_view", "search", "add_to_cart",
                "remove_from_cart", "wishlist_add"]
ev_type = rng.choice(BROWSE_TYPES, N_BROWSE, p=[.29, .385, .125, .105, .04, .055])

# product_id is only meaningful for product-level interactions
needs_prod = np.isin(ev_type, ["product_view", "add_to_cart", "remove_from_cart", "wishlist_add"])
ev_prod_pos = rng.choice(N_PRODUCTS, size=N_BROWSE, p=prod_pop)
ev_prod = np.where(needs_prod, products.product_id.values[ev_prod_pos], None)

# device: usually the customer's preferred one, occasionally another
ev_device = customers.preferred_device.values[ev_cust_pos].copy()
switch = rng.random(N_BROWSE) < 0.18
ev_device[switch] = rng.choice(DEVICES, switch.sum(), p=DEVICE_W)

TRAFFIC = ["organic", "paid_social", "referral", "email", "direct", "affiliate", "push_notification"]
CH2TRAF = {"Organic Search": "organic", "Paid Social": "paid_social", "Referral": "referral",
           "Email Campaign": "email", "Influencer": "affiliate", "Direct": "direct",
           "Marketplace": "affiliate"}
ev_traffic = np.array([CH2TRAF[c] for c in customers.acquisition_channel.values[ev_cust_pos]])
drift = rng.random(N_BROWSE) < 0.35
ev_traffic[drift] = rng.choice(TRAFFIC, drift.sum(), p=[.3, .18, .08, .1, .22, .06, .06])

dwell = np.round(rng.lognormal(3.05, 0.95, N_BROWSE) *
                 np.where(ev_type == "product_view", 1.6,
                          np.where(ev_type == "page_view", 0.7, 1.0)), 1)

events_browse = pd.DataFrame({
    "customer_id": customers.customer_id.values[ev_cust_pos],
    "event_timestamp": ev_ts,
    "event_type": ev_type,
    "product_id": ev_prod,
    "order_id": None,
    "device_type": ev_device,
    "browser": rng.choice(["Chrome Mobile", "Chrome", "Safari", "Samsung Internet",
                           "Firefox", "Opera Mini", "Edge"], N_BROWSE,
                          p=[.42, .18, .12, .11, .05, .08, .04]),
    "traffic_source": ev_traffic,
    "dwell_time_seconds": dwell,
})

# funnel-completing events derived from real orders 
chk = pd.DataFrame({
    "customer_id": orders.customer_id.values,
    "event_timestamp": order_ts - pd.to_timedelta(rng.integers(60, 900, N_ORDERS), unit="s"),
    "event_type": "checkout_start",
    "product_id": None,
    "order_id": orders.order_id.values,
    "device_type": customers.preferred_device.values[order_cust_pos],
    "browser": rng.choice(["Chrome Mobile", "Chrome", "Safari", "Samsung Internet",
                           "Firefox", "Opera Mini", "Edge"], N_ORDERS,
                          p=[.42, .18, .12, .11, .05, .08, .04]),
    "traffic_source": rng.choice(TRAFFIC, N_ORDERS, p=[.3, .18, .08, .1, .22, .06, .06]),
    "dwell_time_seconds": np.round(rng.lognormal(3.8, 0.6, N_ORDERS), 1),
})
pur = pd.DataFrame({
    "customer_id": orders.customer_id.values[item_order_pos],
    "event_timestamp": order_ts.values[item_order_pos],
    "event_type": "purchase",
    "product_id": order_items.product_id.values,
    "order_id": order_items.order_id.values,
    "device_type": customers.preferred_device.values[item_cust_pos],
    "browser": rng.choice(["Chrome Mobile", "Chrome", "Safari", "Samsung Internet",
                           "Firefox", "Opera Mini", "Edge"], N_ITEMS,
                          p=[.42, .18, .12, .11, .05, .08, .04]),
    "traffic_source": rng.choice(TRAFFIC, N_ITEMS, p=[.3, .18, .08, .1, .22, .06, .06]),
    "dwell_time_seconds": np.nan,
})

events = pd.concat([events_browse, chk, pur], ignore_index=True)
events = events.sort_values("event_timestamp", kind="mergesort").reset_index(drop=True)

# session_id: same customer + same calendar day + one of a few visit blocks
day_str = events.event_timestamp.dt.strftime("%y%m%d").values
blk = np.where(events.event_timestamp.dt.hour.values < 12, 1,
               np.where(events.event_timestamp.dt.hour.values < 18, 2, 3))
events["session_id"] = ["SESS-" + c[4:] + d + str(b)
                        for c, d, b in zip(events.customer_id.values, day_str, blk)]
sess_size = events.groupby("session_id", sort=False).customer_id.transform("size")
events["is_bounce"] = (sess_size == 1)
events["event_timestamp"] = events.event_timestamp.dt.strftime("%Y-%m-%d %H:%M:%S")
events.insert(0, "event_id", [f"EVT{i + 1:08d}" for i in range(len(events))])
events = events[["event_id", "session_id", "customer_id", "event_timestamp", "event_type",
                 "product_id", "order_id", "device_type", "browser", "traffic_source",
                 "dwell_time_seconds", "is_bounce"]]


# 6. REVIEWS

print("Generating reviews ...")

delivered = orders.order_status.values == "delivered"
elig = np.where(delivered[item_order_pos])[0]
rev_pos = rng.choice(elig, size=int(len(elig) * 0.21), replace=False)
N_REV = len(rev_pos)

base_rating = products.avg_rating.values[prod_pos[rev_pos]]
late = np.nan_to_num(orders.delivery_days.values[item_order_pos[rev_pos]], nan=3.0)
rating = np.clip(np.round(rng.normal(base_rating, 0.75) - 0.11 * np.maximum(late - 5, 0)), 1, 5)

TITLES = ["Very good", "Value for money", "Not as described", "Delivery was slow",
          "Excellent quality", "Would buy again", "Average", "Damaged on arrival",
          "Fast delivery", "Highly recommend", "Poor packaging", "Exactly what I wanted"]
BODIES = ["Product works as expected, no complaints so far.",
          "Quality is decent for the price paid.",
          "Item arrived later than the estimated date.",
          "Colour is slightly different from the pictures.",
          "Rider was polite and the package was sealed.",
          "Stopped working after two weeks of use.",
          "Good value, I have ordered a second one.",
          "Packaging was torn but the item is fine.",
          "Setup was straightforward and it works well.",
          "Customer service resolved my issue quickly."]

reviews = pd.DataFrame({
    "review_id": [f"REV{i + 1:07d}" for i in range(N_REV)],
    "order_id": order_items.order_id.values[rev_pos],
    "customer_id": customers.customer_id.values[item_cust_pos[rev_pos]],
    "product_id": order_items.product_id.values[rev_pos],
    "review_timestamp": (order_ts.values[item_order_pos[rev_pos]]
                         + pd.to_timedelta(late + rng.integers(1, 30, N_REV), unit="D")),
    "rating": rating.astype(int),
    "review_title": rng.choice(TITLES, N_REV),
    "review_text": rng.choice(BODIES, N_REV),
    "helpful_votes": rng.poisson(1.6, N_REV),
    "verified_purchase": True,
})
reviews["review_timestamp"] = reviews.review_timestamp.dt.strftime("%Y-%m-%d %H:%M:%S")



# 7. DELIBERATE DATA-QUALITY DEFECTS

# Order matters: reformat clean values first, then blank them, then poison with
# sentinels. Everything below is counted in `defects` and printed at the end.
print("Injecting data-quality defects ...")

# customers 
customers["date_of_birth"] = messy_dates(customers.date_of_birth, .12, .07, rng,
                                         "mixed_date_format::customers.date_of_birth")
customers["signup_date"] = messy_dates(customers.signup_date, .10, .06, rng,
                                       "mixed_date_format::customers.signup_date")

# inconsistent categorical encodings
gi = pick_idx(len(customers), .18, rng)
customers["gender"] = customers.gender.astype(object)
customers.loc[customers.index[gi], "gender"] = [
    {"Male": rng.choice(["male", "MALE", "M", " Male"]),
     "Female": rng.choice(["female", "FEMALE", "F", "Female "]),
     "Other": rng.choice(["other", "O", "Prefer not to say"])}[g]
    for g in customers.gender.iloc[gi]]
log("inconsistent_category::customers.gender", len(gi))

ci = pick_idx(len(customers), .10, rng)
TYPOS = {"Accra": "Acrra", "Kumasi": "Kumassi", "Takoradi": "Takoradii",
         "Tamale": "Tamalee", "Tema": "Tem a", "Cape Coast": "Cape coast"}
customers["city"] = customers.city.astype(object)
customers.loc[customers.index[ci], "city"] = [
    rng.choice([c.lower(), c.upper(), f" {c} ", TYPOS.get(c, c)])
    for c in customers.city.iloc[ci]]
log("inconsistent_category::customers.city", len(ci))

ni = pick_idx(len(customers), .10, rng)
customers["full_name"] = customers.full_name.astype(object)
customers.loc[customers.index[ni], "full_name"] = [
    rng.choice([f"  {n}", f"{n}  ", n.upper(), n.lower()]) for n in customers.full_name.iloc[ni]]
log("whitespace_or_case::customers.full_name", len(ni))

ei = pick_idx(len(customers), .05, rng)
customers["email"] = customers.email.astype(object)
customers.loc[customers.index[ei], "email"] = [e.upper() for e in customers.email.iloc[ei]]
log("case_variant::customers.email", len(ei))
sprinkle_values(customers, "email", .006, rng,
                ["not.an.email", "missing@", "@gmail.com", "n/a"],
                "malformed::customers.email")

# messy booleans
bi = pick_idx(len(customers), .26, rng)
customers["email_subscribed"] = customers.email_subscribed.astype(object)
customers.loc[customers.index[bi], "email_subscribed"] = [
    (rng.choice(["yes", "Y", "TRUE", "1"]) if v is True else rng.choice(["no", "N", "FALSE", "0"]))
    for v in customers.email_subscribed.iloc[bi]]
log("mixed_boolean::customers.email_subscribed", len(bi))

customers["annual_income_ghs"] = money_strings(customers.annual_income_ghs, .20, rng,
                                               "currency_string::customers.annual_income_ghs")

# impossible / out-of-range values
sprinkle_values(customers, "date_of_birth", .006, rng,
                ["1899-01-01", "2027-05-10", "0000-00-00"], "impossible_value::customers.date_of_birth")
sprinkle_values(customers, "signup_date", .004, rng,
                ["2027-01-15", "2030-08-01"], "future_date::customers.signup_date")
sprinkle_values(customers, "annual_income_ghs", .02, rng, [-999, -1, 0],
                "sentinel_value::customers.annual_income_ghs")
sprinkle_values(customers, "household_size", .015, rng, [0, -1, 99],
                "impossible_value::customers.household_size")
sprinkle_values(customers, "acquisition_channel", .03, rng, ["unknown", "N/A", ""],
                "placeholder_string::customers.acquisition_channel")
sprinkle_values(customers, "city", .02, rng, ["N/A", "null", "?"],
                "placeholder_string::customers.city")

# genuine nulls
for col, frac in [("gender", .062), ("date_of_birth", .048), ("city", .040),
                  ("region", .035), ("signup_date", .028), ("acquisition_channel", .070),
                  ("preferred_device", .022), ("email_subscribed", .040),
                  ("loyalty_tier", .085), ("annual_income_ghs", .090),
                  ("household_size", .033), ("email", .020)]:
    sprinkle_nulls(customers, col, frac, rng, f"null::customers.{col}")

# near-duplicate customers: same human, brand-new customer_id, cosmetic differences
dup_src = rng.choice(len(customers), 300, replace=False)
near = customers.iloc[dup_src].copy()
near["customer_id"] = [f"CUST9{i:05d}" for i in range(300)]
near["email"] = [str(e).strip().upper() if pd.notna(e) else e for e in near.email]
near["full_name"] = [f" {str(n).title()} " if pd.notna(n) else n for n in near.full_name]
customers = pd.concat([customers, near], ignore_index=True)
log("near_duplicate_person::customers", 300)

customers = duplicate_rows(customers, .020, rng, "exact_duplicate_row::customers")

#  products 
products["launch_date"] = messy_dates(products.launch_date, .10, .05, rng,
                                      "mixed_date_format::products.launch_date")
products["unit_price_ghs"] = money_strings(products.unit_price_ghs, .10, rng,
                                           "currency_string::products.unit_price_ghs")
sprinkle_values(products, "unit_price_ghs", .006, rng, [-49.99, -1, 0],
                "impossible_value::products.unit_price_ghs")
sprinkle_values(products, "stock_quantity", .010, rng, [-5, -20],
                "impossible_value::products.stock_quantity")
pci = pick_idx(len(products), .03, rng)
products["category"] = products.category.astype(object)
products.loc[products.index[pci], "category"] = [
    rng.choice([c.lower(), c.upper(), f"{c} "]) for c in products.category.iloc[pci]]
log("inconsistent_category::products.category", len(pci))
for col, frac in [("unit_price_ghs", .020), ("cost_price_ghs", .030),
                  ("avg_rating", .060), ("brand", .025), ("subcategory", .018)]:
    sprinkle_nulls(products, col, frac, rng, f"null::products.{col}")

# same product_id listed twice with a different price -> conflicting duplicates
conf = products.sample(40, random_state=SEED).copy()
conf["unit_price_ghs"] = [round(float(p) * 1.15, 2) if str(p).replace('.', '', 1).lstrip('-').isdigit()
                          else 199.99 for p in conf.unit_price_ghs]
products = pd.concat([products, conf], ignore_index=True)
log("conflicting_duplicate_key::products.product_id", 40)
products = duplicate_rows(products, .010, rng, "exact_duplicate_row::products")

# orders 
PAY_VARIANTS = {"MTN Mobile Money": ["MTN MoMo", "momo", "mtn mobile money", "MOMO"],
                "Telecel Cash": ["Telecel cash", "telecel", "TELECEL CASH"],
                "AirtelTigo Money": ["AT Money", "airteltigo money", "ATMoney"],
                "Debit Card": ["debit card", "CARD", "Card"],
                "Bank Transfer": ["bank transfer", "BANK TRANSFER", "transfer"],
                "Cash on Delivery": ["COD", "cash on delivery", "Cash On Delivery"]}
pi = pick_idx(len(orders), .13, rng)
orders["payment_method"] = orders.payment_method.astype(object)
orders.loc[orders.index[pi], "payment_method"] = [
    rng.choice(PAY_VARIANTS[p]) for p in orders.payment_method.iloc[pi]]
log("inconsistent_category::orders.payment_method", len(pi))

si = pick_idx(len(orders), .085, rng)
orders["order_status"] = orders.order_status.astype(object)
orders.loc[orders.index[si], "order_status"] = [
    rng.choice([s.capitalize(), s.upper(), f"{s} ", f" {s}"]) for s in orders.order_status.iloc[si]]
log("inconsistent_category::orders.order_status", len(si))

# a slice of timestamps in a different format
ti = pick_idx(len(orders), .03, rng)
orders["order_timestamp"] = orders.order_timestamp.astype(object)
orders.loc[orders.index[ti], "order_timestamp"] = pd.to_datetime(
    orders.order_timestamp.iloc[ti]).dt.strftime("%d/%m/%Y %H:%M")
log("mixed_date_format::orders.order_timestamp", len(ti))

orders["total_amount_ghs"] = money_strings(orders.total_amount_ghs, .15, rng,
                                           "currency_string::orders.total_amount_ghs")
sprinkle_values(orders, "delivery_days", .005, rng, [-2, -1, 365],
                "impossible_value::orders.delivery_days")
for col, frac in [("payment_method", .032), ("shipping_city", .030),
                  ("total_amount_ghs", .020), ("shipping_fee_ghs", .015),
                  ("order_status", .012), ("order_timestamp", .006)]:
    sprinkle_nulls(orders, col, frac, rng, f"null::orders.{col}")

# logical violation: order placed before the customer ever signed up
viol = rng.choice(len(orders), 400, replace=False)
orders.loc[orders.index[viol], "order_timestamp"] = [
    (WINDOW_START - pd.Timedelta(days=int(d))).strftime("%Y-%m-%d %H:%M:%S")
    for d in rng.integers(1000, 1600, 400)]
log("logical_violation::order_before_signup", 400)

# same order_id, two different totals
oconf = orders.sample(250, random_state=SEED).copy()
oconf["total_amount_ghs"] = [round(float(v) * 1.08, 2) if isinstance(v, (int, float)) and pd.notna(v)
                             else v for v in oconf.total_amount_ghs]
orders = pd.concat([orders, oconf], ignore_index=True)
log("conflicting_duplicate_key::orders.order_id", 250)
orders = duplicate_rows(orders, .012, rng, "exact_duplicate_row::orders")

# order_items 
sprinkle_values(order_items, "quantity", .005, rng, [0], "impossible_value::order_items.quantity")
sprinkle_values(order_items, "quantity", .004, rng, [-1, -3], "negative_value::order_items.quantity")
sprinkle_values(order_items, "quantity", .0005, rng, [9999, 5000], "outlier::order_items.quantity")
sprinkle_values(order_items, "discount_pct", .002, rng, [120, 150, -10],
                "impossible_value::order_items.discount_pct")
for col, frac in [("unit_price_ghs", .025), ("discount_pct", .030), ("line_total_ghs", .040)]:
    sprinkle_nulls(order_items, col, frac, rng, f"null::order_items.{col}")

# orphan line items pointing at order_ids that do not exist
orph = order_items.sample(150, random_state=SEED).copy()
orph["order_item_id"] = [f"ITM999{i:05d}" for i in range(150)]
orph["order_id"] = [f"ORD9{i:06d}" for i in range(150)]
order_items = pd.concat([order_items, orph], ignore_index=True)
log("orphan_foreign_key::order_items.order_id", 150)
order_items = duplicate_rows(order_items, .015, rng, "exact_duplicate_row::order_items")

#  events 
sprinkle_values(events, "dwell_time_seconds", .004, rng, [-1, -12.5],
                "negative_value::events.dwell_time_seconds")
sprinkle_values(events, "dwell_time_seconds", .0015, rng, [99999, 86400],
                "outlier::events.dwell_time_seconds")
sprinkle_values(events, "traffic_source", .015, rng, ["", "N/A", "unknown"],
                "placeholder_string::events.traffic_source")
eti = pick_idx(len(events), .020, rng)
events["event_type"] = events.event_type.astype(object)
events.loc[events.index[eti], "event_type"] = [
    rng.choice([t.upper(), t.capitalize(), f"{t} "]) for t in events.event_type.iloc[eti]]
log("inconsistent_category::events.event_type", len(eti))

evti = pick_idx(len(events), .020, rng)
events["event_timestamp"] = events.event_timestamp.astype(object)
events.loc[events.index[evti], "event_timestamp"] = pd.to_datetime(
    events.event_timestamp.iloc[evti]).dt.strftime("%d/%m/%Y %H:%M:%S")
log("mixed_date_format::events.event_timestamp", len(evti))

for col, frac in [("dwell_time_seconds", .040), ("device_type", .025),
                  ("session_id", .012), ("customer_id", .018),
                  ("event_timestamp", .003), ("browser", .020)]:
    sprinkle_nulls(events, col, frac, rng, f"null::events.{col}")

events = duplicate_rows(events, .016, rng, "exact_duplicate_row::events")

# reviews 
sprinkle_values(reviews, "rating", .008, rng, [0, 6, -1], "out_of_range::reviews.rating")
sprinkle_values(reviews, "review_text", .040, rng, ["", "   "], "empty_string::reviews.review_text")
sprinkle_values(reviews, "review_text", .020, rng, ["N/A", "n/a", "null"],
                "placeholder_string::reviews.review_text")
for col, frac in [("rating", .070), ("review_text", .030), ("review_title", .050),
                  ("helpful_votes", .025), ("review_timestamp", .010)]:
    sprinkle_nulls(reviews, col, frac, rng, f"null::reviews.{col}")
reviews = duplicate_rows(reviews, .012, rng, "exact_duplicate_row::reviews")


# 8. WRITE OUTPUT

print("Writing files ...")

tables = {"customers": customers, "products": products, "orders": orders,
          "order_items": order_items, "events": events, "reviews": reviews}
for name, df in tables.items():
    path = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  {name + '.csv':<20} {len(df):>8,} rows  {os.path.getsize(path) / 1e6:>7.1f} MB")

# A JSONL slice of the clickstream, ready to replay into Kafka / Elasticsearch
stream = events.head(40_000).where(pd.notna(events.head(40_000)), None)
with open(os.path.join(OUT_DIR, "events_stream_sample.jsonl"), "w") as fh:
    for rec in stream.to_dict(orient="records"):
        fh.write(json.dumps(rec, default=str) + "\n")
print(f"  {'events_stream_sample.jsonl':<20} 40,000 lines")


# 9. VERIFICATION LOG

print("\n" + "=" * 78)
print("DEFECT LOG  (use this to check your cleaning caught everything)")
print("=" * 78)
for k in sorted(defects):
    print(f"  {k:<58} {defects[k]:>9,}")
print(f"\n  {'TOTAL DEFECTS INJECTED':<58} {sum(defects.values()):>9,}")

print("\n" + "=" * 78)
print("TABLE SUMMARY")
print("=" * 78)
for name, df in tables.items():
    full_dupes = df.duplicated().sum()
    nulls = df.isna().sum().sum()
    print(f"\n{name}: {len(df):,} rows x {df.shape[1]} cols | "
          f"{full_dupes:,} exact duplicate rows | {nulls:,} null cells "
          f"({100 * nulls / (len(df) * df.shape[1]):.1f}% of cells)")
    nn = df.isna().sum()
    nn = nn[nn > 0].sort_values(ascending=False)
    for c, v in nn.items():
        print(f"    {c:<26} {v:>8,} nulls ({100 * v / len(df):.1f}%)")

# Hidden ground truth, written separately so it cannot leak into the features.
truth = pd.DataFrame({
    "customer_id": cust_ids,
    "latent_engagement": engagement.round(4),
    "latent_discount_affinity": discount_affinity.round(4),
    "churned_by_design": will_churn,
    "last_active_date": active_end.dt.date.astype(str),
})
truth.to_csv(os.path.join(OUT_DIR, "_hidden_ground_truth.csv"), index=False)
print(f"\nGround truth written for {len(truth):,} customers "
      f"({100 * will_churn.mean():.1f}% churned by design). "
      "Do NOT use as a model feature.")
print("\nDone.")