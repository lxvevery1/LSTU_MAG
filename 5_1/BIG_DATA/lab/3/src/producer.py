from kafka import KafkaProducer
import json
import time
from datetime import datetime

# Настройки подключения
KAFKA_BROKER = 'localhost:9092'
TOPIC_NAME = 'test-topic'

# Создание продюсера
producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

def send_message(message):
    """Отправка сообщения в Kafka"""
    try:
        # Добавляем временную метку
        message['timestamp'] = datetime.now().isoformat()

        # Отправляем сообщение
        producer.send(TOPIC_NAME, value=message)
        producer.flush()

        print(f"Отправлено сообщение: {message}")
        return True
    except Exception as e:
        print(f"Ошибка при отправке: {e}")
        return False

if __name__ == "__main__":
    print("Запуск продюсера Kafka...")
    print("Для остановки нажмите Ctrl+C")

    try:
        counter = 0
        while True:
            # Формируем тестовое сообщение
            message = {
                'id': counter,
                'message': f'Тестовое сообщение #{counter}',
                'source': 'python-producer'
            }

            # Отправляем сообщение
            send_message(message)

            # Увеличиваем счетчик и ждем
            counter += 1
            time.sleep(2)

    except KeyboardInterrupt:
        print("Остановка продюсера...")
    finally:
        producer.close()
