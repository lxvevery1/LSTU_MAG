# TASK 3
# meanAir.py
# Напишите функцию, которая рассчитывает среднее значение по одному из столбцов factor (Ozone,
# Solar.R, Wind), когда Temp принимает значения от tMin до tMax включительно. Значения по-умолчанию
# tMin = 60, tMax = 80

import pandas as pd

ozone_name = 'Ozone'
solar_name = 'Solar.R'
wind_name = 'Wind'
temp_name = 'Temp'
month_name = 'Month'
day_name = 'Day'

tMin = 60
tMax = 80

wMin = 8
wMax = 13
def calculate_mean_wind(factor, wMin, wMax):
    """
    Рассчитывает среднее значение по столбцу factor для строк,
    где температура T находится в диапазоне [tmin, tmax]

    Parameters:
    factor (str): название столбца
    tmin (float): минимальное значение температуры (по умолчанию 60)
    tmax (float): максимальное значение температуры (по умолчанию 80)

    Returns:
    float: среднее значение указанного фактора
    """
    try:
        df = pd.read_csv('dataset.csv')

        # wMin < Wind <= wMax
        filtered_data = df[(df[wind_name] > wMin) & (df[wind_name] <= wMax)]

        print(filtered_data)

        if len(filtered_data) == 0:
            raise ValueError(f"Нет данных с ветром {factor} в диапазоне [{wMin}, {wMax}]")

        mean_value = filtered_data[factor].mean()
        mean_value_1 = filtered_data[factor].std()

        print(f"Среднее значение {factor}: {mean_value:.2f}")
        print(f"Среднее квадратическое отклонение {factor}: {mean_value_1:.2f}")

        return mean_value

    except FileNotFoundError:
        raise FileNotFoundError("Файл dataset.csv не найден. Убедитесь, что он находится в текущей директории.")
    except Exception as e:
        raise Exception(f"Ошибка при расчете: {str(e)}")

def calculate_mean(factor, tmin=tMin, tmax=tMax):
    """
    Рассчитывает среднее значение по столбцу factor для строк,
    где температура T находится в диапазоне [tmin, tmax]

    Parameters:
    factor (str): название столбца
    tmin (float): минимальное значение температуры (по умолчанию 60)
    tmax (float): максимальное значение температуры (по умолчанию 80)

    Returns:
    float: среднее значение указанного фактора
    """
    try:
        df = pd.read_csv('dataset.csv')

        filtered_data = df[(df[temp_name] >= tmin) & (df[temp_name] <= tmax)]

        if len(filtered_data) == 0:
            raise ValueError(f"Нет данных с температурой в диапазоне [{tmin}, {tmax}]")

        mean_value = filtered_data[factor].mean()

        count = len(filtered_data)
        print(f"Найдено {count} строк с ветром от {tmin} до {tmax}°C")
        print(f"Среднее значение {factor}: {mean_value:.2f}")

        print(f"df[")

        return mean_value

    except FileNotFoundError:
        raise FileNotFoundError("Файл dataset.csv не найден. Убедитесь, что он находится в текущей директории.")
    except Exception as e:
        raise Exception(f"Ошибка при расчете: {str(e)}")

if __name__ == "__main__":
    try:
        result = calculate_mean_wind(solar_name, wMin, wMax)
        print(f"Результат: {result:.2f}")
    except Exception as e:
        print(f"Ошибка: {e}")
