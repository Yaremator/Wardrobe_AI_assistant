# Wardrobie - Smart Wardrobe Assistant

## Запуск застосунку

### Активувати віртуальне середовище та запустити
```bash
source .venv/bin/activate
streamlit run main.py
```

Застосунок буде доступний за адресою: http://localhost:8501

### Альтернативно (без активації venv)
```bash
.venv/bin/streamlit run main.py
```

## Що було виправлено

Видалено непотрібну subprocess обгортку, яка призводила до запуску кількох екземплярів Streamlit на різних портах (8501, 8502). Тепер [`main.py`](main.py) є чистим точковим входом для Streamlit без рекурсивних викликів.

### Зміни в main.py
- ❌ Видалено `launch_with_streamlit()` функцію
- ❌ Видалено subprocess логіку
- ❌ Видалено змінну `WARDROBIE_RUNNING_IN_STREAMLIT`
- ❌ Видалено умовну логіку `if __name__ == "__main__"`
- ✅ Код винесено на верхній рівень модуля
- ✅ Замінено `return` на `st.stop()` після автентифікації

### Результат
- ✅ Один процес Streamlit
- ✅ Один порт (8501)
- ✅ Стабільна робота
- ✅ Коректні reruns без дублювання

## Детальна документація

Для детального аналізу проблеми та рішення див. [`plans/fix-streamlit-double-instance.md`](plans/fix-streamlit-double-instance.md)
