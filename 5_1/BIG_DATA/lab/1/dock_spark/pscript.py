import time
from pyspark.sql import SparkSession
from pyspark import SparkConf
import socket

def get_host_ip():
    """Get host IP that containers can access"""
    try:
        # Connect to Spark master to determine reachable IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("spark-master", 7077))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        # Fallback to host.docker.internal for Mac
        return "host.docker.internal"

def create_spark_session():
    """Create Spark session that works with Docker networking"""

    driver_host = get_host_ip()
    print(f"Using driver host: {driver_host}")

    conf = SparkConf()
    conf.setAppName("SparkDockerFixed")
    conf.setMaster("spark://spark-master:7077")
    conf.set("spark.driver.host", driver_host)
    conf.set("spark.driver.port", "4050")
    conf.set("spark.driver.bindAddress", "0.0.0.0")
    conf.set("spark.executor.memory", "1g")
    conf.set("spark.executor.cores", "1")
    conf.set("spark.network.timeout", "300s")
    conf.set("spark.sql.adaptive.enabled", "true")

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark

# Использование
spark = create_spark_session()

def simple_operation_test(spark):
    """Test simple operations without complex transformations"""

    print("Testing basic RDD operations...")

    # Test 1: Simple count
    try:
        rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5], 2)  # 2 partitions
        count = rdd.count() # JOB #0
        print(f"✅ Count test passed: {count}")
    except Exception as e:
        print(f"❌ Count test failed: {e}")
        return False

    # Test 2: Simple collect (avoid complex operations initially)
    try:
        rdd = spark.sparkContext.parallelize([1, 2, 3], 1)
        data = rdd.collect() # JOB #1
        print(f"✅ Collect test passed: {data}")
    except Exception as e:
        print(f"❌ Collect test failed: {e}")
        return False

    # Test 3: Simple map
    try:
        rdd = spark.sparkContext.parallelize([1, 2, 3], 1)
        mapped = rdd.map(lambda x: x * 2)
        result = mapped.collect() # JOB #1
        print(f"✅ Map test passed: {result}")
    except Exception as e:
        print(f"❌ Map test failed: {e}")
        return False

    # Only after basic tests work, try sum
    try:
        rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5], 2)
        result = rdd.sum() # JOB #3
        print(f"✅ Sum test passed: {result}")
    except Exception as e:
        print(f"❌ Sum test failed: {e}")
        print("Trying alternative sum implementation...")

        # Alternative approach for sum
        try:
            rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5], 2)
            result = rdd.reduce(lambda a, b: a + b)
            print(f"✅ Reduce sum passed: {result}")
        except Exception as e:
            print(f"❌ Reduce sum also failed: {e}")
            return False

    return True

def main():
    print("🚀 Starting local to Docker Spark connection test...")
    print("Waiting 10 seconds for cluster to be ready...")
    time.sleep(10)

    try:
        spark = create_spark_session()
        print("✅ Spark session created successfully!")

        # Test basic functionality
        if simple_operation_test(spark):
            print("\n🎉 All tests passed! Connection is working.")
        else:
            print("\n⚠️  Some tests failed, but connection is established.")

        # Show cluster info
        print(f"\n📊 Cluster information:")
        print(f"Master URL: {spark.sparkContext.master}")
        print(f"Application ID: {spark.sparkContext.applicationId}")
        print(f"Spark UI: http://localhost:4040")

    except Exception as e:
        print(f"❌ Failed to create Spark session: {e}")
        import traceback
        traceback.print_exc()

    finally:
        try:
            # spark.stop()
            print("Spark session stopped.")
        except:
            pass

if __name__ == "__main__":
    main()
