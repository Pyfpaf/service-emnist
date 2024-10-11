# Распознавание рукописных символов EMNIST

Данный репозиторий представляет собой сервис по распознаванию единичных английских рукописных символов и цифр.
## 1. Описание решения

Решение задачи распознавания символов было реализовано на нейронной сети с помощью фреймворка PyTorch.
В качестве обучающей и валидационной выборок была использована база рукописных символов EMNIST.

Для работы с сервисом необходимо в адресной строке браузера ввести localhost:8000, после чего откроется окно с областью по написанию символа с помощью мышки.
После написания символа необходимо нажать на кнопку "Predict". В качестве ответа сервис выведет символ, который он распознал.
Очистка области написания символа происходит с помощью кнокпи "Clear".


## 2. Установка и запуск сервиса

Для запуска сервиса в локальном docker контейнере необходимо произвести следующие действия:

1. Клонируем глобальный репозиторий Git на локальный компьютер с помощью команды

git clone https://github.com/Pyfpaf/service-emnist.git

2. Переходим в папку репозитория с помощью команды

cd service-emnist

3. Проверяем запущен-ли на локальном компьютере сервис docker после чего выполняем команду для создания docker образа

docker build -t service-emnist .
(Точка в конце команды - ее обязательный элемент)

4. Запускаем docker контейнер из созданного образа с помощью команды

docker run -p 8000:8000 service-emnist

5. В адресной строке браузера вводим localhost:8000 после чего должно открыться окно с областью написания символов.

