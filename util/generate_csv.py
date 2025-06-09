import pandas as pd
import random
from datetime import timedelta, datetime

locations = ['Pune', 'Mumbai', 'Nagpur', 'Nashik', 'Aurangabad', 'Thane']
types = ['Truck', 'Storage', 'Packaging']
companies = [f'Company {i}' for i in range(1, 21)]

data = []
for i in range(1000):
    loc_from = random.choice(locations)
    loc_to = random.choice([l for l in locations if l != loc_from])
    desc = f"{random.randint(5, 30)} tons {random.choice(['truck', 'container'])} space from {loc_from} to {loc_to}"
    data.append({
        'company': random.choice(companies),
        'description': desc,
        'location_from': loc_from,
        'location_to': loc_to,
        'type_of_capacity': random.choice(types),
        'available_till': (datetime.today() + timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d')
    })

df = pd.DataFrame(data)
df.to_csv('offers.csv', index=False)
print("Generated offers.csv")