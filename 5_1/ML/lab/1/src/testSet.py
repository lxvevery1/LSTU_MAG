# TASK 6
# testSet.py
# Реализуйте функцию testSet(perc = 20), которая возвращает тестовое множество, состоящее из
# заданного прцоента строк. Строки должны выбираться случайным образом. Параметр perc -
# вещественное число от 0 до 100. Если переданное значение параметра выхдит за пределы [0;100]
# неободимо выдать сообщение об ошибке.

import pandas as pd
import random

ozone_name = 'Ozone'
solar_name = 'Solar.R'
wind_name = 'Wind'
temp_name = 'Temp'
month_name = 'Month'
day_name = 'Day'

def testSet(perc=20):
    """
    Возвращает множество, состоящее из заданного процента строк, выбранных случайно

    Parameters:
    perc (int): процент строк для выборки (от 0 до 100, по умолчанию 20)

    Returns:
    set: множество индексов выбранных строк
    """
    try:
        if not 0 <= perc <= 100:
            raise ValueError("Параметр perc должен быть в диапазоне от 0 до 100")

        df = pd.read_csv('dataset.csv')

        total_rows = len(df)

        sample_size = int(total_rows * perc / 100)

        if perc > 0 and sample_size == 0:
            sample_size = 1

        random_indices = sorted(set(random.sample(range(total_rows), sample_size)))

        print(f"Всего строк в датасете: {total_rows}")
        print(f"Выбрано {len(random_indices)} строк ({perc}%)")
        print(f"Индексы выбранных строк: {sorted(random_indices)}")

        print(df.loc[list(random_indices)])

        return random_indices

    except FileNotFoundError:
        raise FileNotFoundError("Файл dataset.csv не найден. Убедитесь, что он находится в текущей директории.")
    except ValueError as ve:
        raise ve
    except Exception as e:
        raise Exception(f"Ошибка при выполнении функции: {str(e)}")

if __name__ == "__main__":
    try:
        print("Пример 1: 20% строк (по умолчанию)")
        result1 = testSet()
        print(f"Выбрано {len(result1)} индексов\n")

    except ValueError as ve:
        print(f"Ошибка в параметрах: {ve}")
    except Exception as e:
        print(f"Ошибка: {e}")
