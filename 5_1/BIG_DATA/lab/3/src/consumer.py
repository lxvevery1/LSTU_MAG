from kafka import KafkaConsumer
import json

# Настройки подключения
KAFKA_BROKER = 'localhost:9092'
TOPIC_NAME = 'test-topic'

# Создание консьюмера
consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=[KAFKA_BROKER],
    auto_offset_reset='earliest',  # начинать чтение с начала топика
    enable_auto_commit=True,
    group_id='python-consumer-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

def consume_messages():
    """Чтение сообщений из Kafka"""
    print("Запуск консьюмера Kafka...")
    print("Ожидание сообщений. Для остановки нажмите Ctrl+C")

    try:
        for message in consumer:
            value = message.value
            print(f"Получено сообщение: {value}")
            print(f"Топик: {message.topic}, Партиция: {message.partition}")
            print(f"Offset: {message.offset}, Timestamp: {message.timestamp}")
            print("-" * 50)

    except KeyboardInterrupt:
        print("Остановка консьюмера...")
    finally:
        consumer.close()

if __name__ == "__main__":
    consume_messages()
