import pandas as pd
import numpy as np
from datetime import datetime
import random

# Set random seed for reproducibility
np.random.seed(42)

# Number of entries
n_entries = 10000

# Generate data
current_year = datetime.now().year

# Define possible values for categorical features
brands = ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Audi', 'Volkswagen', 'Hyundai', 'Kia', 'Nissan']
models = {
    'Toyota': ['Camry', 'Corolla', 'RAV4', 'Highlander', 'Prius'],
    'Honda': ['Civic', 'Accord', 'CR-V', 'Pilot', 'HR-V'],
    'Ford': ['F-150', 'Escape', 'Explorer', 'Mustang', 'Focus'],
    'BMW': ['3 Series', '5 Series', 'X3', 'X5', '7 Series'],
    'Mercedes': ['C-Class', 'E-Class', 'GLC', 'GLE', 'S-Class'],
    'Audi': ['A3', 'A4', 'Q5', 'Q7', 'A6'],
    'Volkswagen': ['Golf', 'Passat', 'Tiguan', 'Atlas', 'Jetta'],
    'Hyundai': ['Elantra', 'Sonata', 'Tucson', 'Santa Fe', 'Kona'],
    'Kia': ['Forte', 'Optima', 'Sportage', 'Telluride', 'Sorento'],
    'Nissan': ['Altima', 'Sentra', 'Rogue', 'Maxima', 'Pathfinder']
}
fuel_types = ['Gasoline', 'Diesel', 'Hybrid', 'Electric', 'Plugin Hybrid']
transmission_types = ['Automatic', 'Manual', 'CVT', 'DCT']

# Generate random data
data = {
    'Year': np.random.randint(2010, current_year + 1, n_entries),
    'Brand': np.random.choice(brands, n_entries),
    'Mileage': np.random.randint(0, 150000, n_entries),
    'Engine_Size': np.random.uniform(1.0, 6.0, n_entries).round(1),
    'Horsepower': np.random.randint(100, 500, n_entries),
    'Fuel_Type': np.random.choice(fuel_types, n_entries),
    'Transmission_Type': np.random.choice(transmission_types, n_entries),
    'Price': np.zeros(n_entries)  # Will be calculated based on features
}

# Add models based on brands
data['Model'] = [np.random.choice(models[brand]) for brand in data['Brand']]

# Create DataFrame
df = pd.DataFrame(data)

# Calculate realistic prices based on features
def calculate_price(row):
    base_price = 20000
    
    # Age depreciation
    age = current_year - row['Year']
    age_factor = 0.93 ** age
    
    # Mileage depreciation
    mileage_factor = 0.9 ** (row['Mileage'] / 20000)
    
    # Brand factor
    brand_factors = {
        'Toyota': 1.0, 'Honda': 1.0,
        'Ford': 0.9,
        'BMW': 1.5, 'Mercedes': 1.6, 'Audi': 1.4,
        'Volkswagen': 0.95,
        'Hyundai': 0.85, 'Kia': 0.85,
        'Nissan': 0.9
    }
    
    # Engine and horsepower factor
    power_factor = (row['Engine_Size'] * 0.1 + row['Horsepower'] * 0.002)
    
    # Fuel type factor
    fuel_factors = {
        'Gasoline': 1.0,
        'Diesel': 1.1,
        'Hybrid': 1.2,
        'Electric': 1.3,
        'Plugin Hybrid': 1.25
    }
    
    final_price = (base_price * 
                  age_factor * 
                  mileage_factor * 
                  brand_factors[row['Brand']] * 
                  power_factor * 
                  fuel_factors[row['Fuel_Type']])
    
    # Add some random variation
    final_price *= np.random.uniform(0.95, 1.05)
    
    return round(final_price, 2)

# Calculate prices
df['Price'] = df.apply(calculate_price, axis=1)

# Add some categorical features for classification
df['Category'] = pd.cut(df['Price'], 
                       bins=[0, 20000, 35000, 50000, 75000, float('inf')],
                       labels=['Budget', 'Economy', 'Mid-Range', 'Luxury', 'Premium'])

# Save to CSV
df.to_csv('car_data.csv', index=False)

print("Dataset generated successfully!")
print(f"Shape of the dataset: {df.shape}")
print("\nSample of the generated data:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())