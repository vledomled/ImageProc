import pandas as pd
from pathlib import Path

# Загрузить таблицу
df = pd.read_excel('6224.xlsx')

name_line = 'loh'

# Убедиться, что первая колонка — число
df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')

# Убираем строки, где в первой колонке NaN
df = df.dropna(subset=[df.columns[0]])

# Задаём диапазон длин волн
lower_bound = 442.0
upper_bound = 443.4

# Фильтруем строки по первой колонке
filtered_df = df[(df.iloc[:, 0] >= lower_bound) & (df.iloc[:, 0] <= upper_bound)]

# Сохраняем результат
filtered_df.to_excel(Path('results') / f'{name_line}_gauss.xlsx', index=False, header= False)

print(f"Фильтрация завершена! Сохранено {len(filtered_df)} строк.")
