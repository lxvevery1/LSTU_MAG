# TASK 5
# maxTemp.py
# Реализуйте функцию maxTemp(days = 1), которая возращает требуемое колическтво пар месяц/день
# с максимальной температурой. Например, если параметр days == 3, то должны быть выведены ровно
# 3 пары значений "месяц/день". Параметр days - кол-во дней - некоторое натуральное число.
import pandas as pd

ozone_name = 'Ozone'
solar_name = 'Solar.R'
wind_name = 'Wind'
temp_name = 'Temp'
month_name = 'Month'
day_name = 'Day'

def maxTemp(n=5):
    """
    Возвращает n пар (месяц, день) с максимальной температурой

    Parameters:
    n (int): количество возвращаемых пар (по умолчанию 5)

    Returns:
    list: список кортежей в формате [(месяц, день, температура), ...]
    """
    try:
        df = pd.read_csv('dataset.csv')

        if month_name not in df.columns or temp_name not in df.columns:
            raise ValueError("В dataset.csv отсутствуют необходимые столбцы {month_name} или {temp_name}")

        daily_max_temp = df.groupby([month_name, day_name])[temp_name].max().reset_index()

        daily_max_temp_sorted = daily_max_temp.sort_values(temp_name, ascending=False)

        top_n_days = daily_max_temp_sorted.head(n)

        print(top_n_days)

        result = zip(
            top_n_days[month_name],
            top_n_days[day_name],
            top_n_days[temp_name]
        )

        return result

    except FileNotFoundError:
        raise FileNotFoundError("Файл dataset.csv не найден. Убедитесь, что он находится в текущей директории.")
    except Exception as e:
        raise Exception(f"Ошибка при выполнении функции: {str(e)}")

def print_max_temp_results(results):
    """
    Красиво выводит результаты функции maxTemp
    """
    month_names = {
        1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
        5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
        9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
    }

    print("Дни с максимальной температурой:")
    print("-" * 40)
    for i, (month, day, temp) in enumerate(results, 1):
        month_name = month_names.get(month, f'Месяц {month}')
        print(f"{i:2}. {month_name:10} {day:2} число: {temp:.1f}°C")

if __name__ == "__main__":
    try:
        results = maxTemp(5)
        print_max_temp_results(results)

    except Exception as e:
        print(f"Ошибка: {e}")
