# import streamlit as st

# Title
#st.title ("Sandbox")

# st.header("Sreamlite lesson")

# st.subheader("Work with text")

# st.text("Any text info")

# Markdown
# st.markdown("### Заголовок третьего уровня")
# st.markdown("*Текст* в **Markdown**!")

#Information masseges
# st.success("Success")
# st.info("Information")
# st.warning("Warning")
# st.error("Error")

# импортируем функцию Image, чтобы открывать картинки
# from PIL import Image
#
# # загружаем картинку
# img = Image.open(r"D:\Users\cspla\Documents\EDA_Cars\streamlit.png")
#
# # отображаем картинку используя streamlit
# st.image(img, width=200)
#
# # проверяем выбран ли чекбокс
# if st.checkbox("Show/Hide"):
#     # показываем текст если чекбокс выбран
#     st.text("Showing the widget")
#
# status = st.radio("Select Gender: ", ('Male', 'Female'))
#
# if (status == 'Male'):
#     st.success("Male")
# else:
#     st.success("Female")

# импортируем библиотеку streamlit
import streamlit as st

# заголовок приложения
st.title('Калькулятор индекса массы тела (ИМТ)')

# СЧИТЫВАЕМ ВЕС
weight = st.number_input("Введите ваш вес (в килограммах)")

# СЧИТЫВАЕМ РОСТ
# используем radio button, чтобы указать единицы измерения
status = st.radio('Укажите единицы измерения роста: ', ('см', 'м', 'футы'))

# сравниваем различные статусы для единиц измерения роста
if (status == 'см'):
    # считываем значение роста в сантиметрах
    height = st.number_input('Сантиметры')

    try:
        bmi = weight / ((height / 100) ** 2)
    except:
        st.text("Введите ваш рост")

elif (status == 'м'):
    # считываем значение роста в метрах
    height = st.number_input('Метры')

    try:
        bmi = weight / (height ** 2)
    except:
        st.text("Введите ваш рост")

else:
    # считываем значение роста в футах
    height = st.number_input('Футы')

    # 1 meter = 3.28
    try:
        bmi = weight / (((height / 3.28)) ** 2)
    except:
        st.text("Введите ваш рост")

# проверяем нажата кнопка или нет
if (st.button('Рассчитать ИМТ')):

    # напечатать значение ИМТ
    st.text(f"Ваш ИМТ равен {bmi:.2f}")

    # интерпретация ИМТ
    if (bmi < 16):
        st.error("Выраженный дефицит массы тела")
    elif (bmi >= 16 and bmi < 18.5):
        st.warning("Недостаточная (дефицит) масса тела")
    elif (bmi >= 18.5 and bmi < 25):
        st.success("Норма")
    elif (bmi >= 25 and bmi < 30):
        st.warning("Избыточная масса тела")
    elif (bmi >= 30):
        st.error("Любитель вкусняшек")
