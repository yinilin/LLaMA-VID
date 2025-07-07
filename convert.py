import tensorflow as tf
import os

def extract_events(file_path, output_path):
    # 创建一个文件来写入提取的数据
    with open(output_path, 'w') as out_file:
        # 创建一个事件文件迭代器
        for event in tf.compat.v1.train.summary_iterator(file_path):
            out_file.write(f"Wall Time: {event.wall_time}\n")
            out_file.write(f"Step: {event.step}\n")
            for value in event.summary.value:
                out_file.write(f"Tag: {value.tag}\n")
                out_file.write(f"Value: {value.simple_value}\n")
            out_file.write("\n")

# 指定事件文件的路径和输出文本文件的路径
event_file_path = 'events.out.tfevents.1751639376.krylov-wf-pod-xo8o8cezusqvulldlynvjxqrhjs0vrvoe-avfraia-m.172.0'
output_txt_path = 'output.txt'

extract_events(event_file_path, output_txt_path)