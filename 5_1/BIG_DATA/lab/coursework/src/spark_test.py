cat << 'EOF' > /tmp/load.py
from pyspark.sql import SparkSession
import time

spark = SparkSession.builder \
    .appName("WorkerLoadTest") \
    .getOrCreate()

sc = spark.sparkContext

# Много партиций → точно на worker
rdd = sc.parallelize(range(0, 200_000_000), 64)

def heavy(x):
    v = x
    for _ in range(100):
        v = (v * 31 + 7) % 1000003
    return v

# CPU + shuffle
result = rdd.map(heavy).reduce(lambda a, b: a + b)

print("Result:", result)

# Держим приложение живым
time.sleep(600)

spark.stop()
EOF
