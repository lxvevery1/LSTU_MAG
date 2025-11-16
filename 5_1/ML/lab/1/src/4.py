# TASK 4
# Найдите и выведите на экран средние значения по столбцу Solar.R для каждого месяца

import pandas as pd

df = pd.read_csv('dataset.csv')

ozone_name = 'Ozone'
solar_name = 'Solar.R'
wind_name = 'Wind'
temp_name = 'Temp'
month_name = 'Month'
day_name = 'Day'

monthly_solar_mean = df.groupby(month_name)[solar_name].mean()

print("Средние значения по столбцу {solar_name} для каждого месяца:")
print("-" * 45)

for month, mean_value in monthly_solar_mean.items():
    print(f"Месяц {month}: {mean_value:.2f}")

print(f"\nОбщая статистика:")
print(f"Среднее {solar_name} за все месяцы: {df[solar_name].mean():.2f}")
print(f"Месяц с максимальным {solar_name}: {monthly_solar_mean.idxmax()} ({monthly_solar_mean.max():.2f})")
print(f"Месяц с минимальным {solar_name}: {monthly_solar_mean.idxmin()} ({monthly_solar_mean.min():.2f})")
