mport sys
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def main(spark, train_file, val_file, test_file, train_new_file, val_new_file, test_new_file):
    train_data = spark.read.parquet(train_file)
    val_data = spark.read.parquet(val_file)
    test_data = spark.read.parque(test_file)
    print('Finishe reading all data')

    train_new = train_data.select(train_data.count > 3)
    val_new = val_data.select(val_data.count > 3 )
    test_new = test_data.select(test_data.count > 3)
    print('Finish dropping the low counts')

    train_new = train_new.repartition(5000, 'user_label')
    val_new = val_new.repartition('user_label')
    test_new = test_new.repartition('user_label')

    train_new.write.parquet(train_new_file)
    print('Finish writing training_data')
    val_new.write.parquet(val_new_file)
    print('Finish writing val_data')
    test_new.write.parquet(test_new_file)
    print('Finish writing test_data')

if __name__ == '__main__':

    conf = SparkConf()
    conf.set('spark.executor.memory', '16g')
    conf.set('spark.driver.memory', '16g')
    conf.set('spark.default.parallelism', '4')

    # create the spark session object
    spark = SparkSession.builder.config(conf = conf).appName('extension_log').getOrCreate()

    # paths for the original files
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    test_file = sys.argv[3]

    # paths to store the transformed files
    train_new_file = sys.argv[4]
    val_new_file = sys.argv[5]
    test_new_file = sys.argv[6]

    #Call the main function
    main(spark, train_file, val_file, test_file, train_new_file,
         val_new_file, test_new_file)

