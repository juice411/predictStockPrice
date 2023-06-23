import tensorflow as tf

# 输出可见的物理GPU设备
physical_devices = tf.config.list_physical_devices('GPU')
print("可见的物理GPU设备：", physical_devices)

# 输出可见的逻辑GPU设备
logical_devices = tf.config.list_logical_devices('GPU')
print("可见的逻辑GPU设备：", logical_devices)
