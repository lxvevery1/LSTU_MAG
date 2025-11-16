# TASK (1=R-oriented) 2
# Посчитайте:
#   * Число строк в таблице
#   * Число столбцов в таблице
#   * Число строк, без NA
#   * Число строк, имеющих пропуски одновременно по столбцам Ozone и Solar.R
#   * Диапазоны варьированя (мин. и макс. значения), а также средние значения по столбцам
#   Ozone, Solar.R, Wind и Temp (без NA)
#   * Среднее значение по столбцу Solar.R для 5-го месяца (без NA)
import pandas as pd

df = pd.read_csv('dataset.csv')

ozone_name = 'Ozone'
solar_name = 'Solar.R'
wind_name = 'Wind'
temp_name = 'Temp'
month_name = 'Month'
day_name = 'Day'

print(f"Число строк в таблице: {df.shape[0]}")
print(f"Число столбцов в таблице: {df.shape[1]}")

no_missing_anywhere = df.dropna().shape[0]
print(f"Число строк без пропусков вообще: {no_missing_anywhere}")

o3_rh_missing = df[df[ozone_name].isnull() & df[solar_name].isnull()].shape[0]
print(f"Число строк с пропусками одновременно по { ozone_name } и { solar_name }: {o3_rh_missing}")

columns_to_analyze = [ozone_name, solar_name, wind_name, temp_name]

print(f"\nСтатистика по столбцам:")
print("-" * 50)

for col in columns_to_analyze:
    if col in df.columns:
        print(f"\nСтолбец: {col}")
        print(f"  Минимальное значение: {df[col].min():.2f}")
        print(f"  Максимальное значение: {df[col].max():.2f}")
        print(f"  Диапазон варьирования: {df[col].max() - df[col].min():.2f}")
        print(f"  Среднее значение: {df[col].mean():.2f}")
        print(f"  Количество пропусков: {df[col].isnull().sum()}")
    else:
        print(f"\nСтолбец {col} не найден в таблице")

solar_mean_5th_month = df.loc[df[month_name] == 5, solar_name].dropna().mean()

print(f"\nСреднее значение по столбцу {solar_name} для 5-го месяца: {solar_mean_5th_month:.2f}")

print(f"\nДополнительная информация:")
print(f"Всего строк с пропусками в любом столбце: {df.isnull().any(axis=1).sum()}")
print(f"Процент строк без пропусков: {no_missing_anywhere/df.shape[0]*100:.2f}%")
