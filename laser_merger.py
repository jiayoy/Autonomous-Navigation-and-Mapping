#!/usr/bin/env python
import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion
import tf2_ros
import tf2_geometry_msgs

class LaserMerger:
    def __init__(self):
        rospy.init_node('laser_merger_node', anonymous=True)
        
        # 参数配置
        self.front_topic = rospy.get_param('~front_topic', '/scan_front')
        self.rear_topic = rospy.get_param('~rear_topic', '/scan_rear')
        self.output_topic = rospy.get_param('~output_topic', '/merged')
        self.output_frame = rospy.get_param('~output_frame', 'base_link')
        
        # TF缓存
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 发布者和订阅者
        self.scan_pub = rospy.Publisher(self.output_topic, LaserScan, queue_size=10)
        rospy.Subscriber(self.front_topic, LaserScan, self.scan_callback, callback_args='front')
        rospy.Subscriber(self.rear_topic, LaserScan, self.scan_callback, callback_args='rear')
        
        # 数据存储
        self.last_scan = {'front': None, 'rear': None}
        self.ready_to_publish = False
        
        rospy.loginfo("激光融合节点启动，输出话题: %s", self.output_topic)
    
    def scan_callback(self, msg, scan_type):
        # 存储当前扫描
        self.last_scan[scan_type] = msg
        
        # 当两个扫描都有数据时触发融合
        if self.last_scan['front'] is not None and self.last_scan['rear'] is not None:
            self.merge_scans()
    
    def merge_scans(self):
        try:
            # 获取当前扫描的时间
            stamp = rospy.Time.now()
            
            # 创建融合扫描消息
            merged = LaserScan()
            merged.header = Header()
            merged.header.stamp = stamp
            merged.header.frame_id = self.output_frame
            
            # 配置扫描参数
            merged.angle_min = -math.pi
            merged.angle_max = math.pi
            merged.angle_increment = 0.01  # 每度约0.01745弧度
            merged.scan_time = max(self.last_scan['front'].scan_time, 
                                  self.last_scan['rear'].scan_time, 0.1)
            merged.time_increment = 0.0001
            merged.range_min = min(self.last_scan['front'].range_min,
                                  self.last_scan['rear'].range_min, 0.1)
            merged.range_max = max(self.last_scan['front'].range_max,
                                  self.last_scan['rear'].range_max, 30.0)
            
            # 计算总点数
            num_points = int((merged.angle_max - merged.angle_min) / 
                            merged.angle_increment) + 1
            merged.ranges = [float('nan')] * num_points
            merged.intensities = [0] * num_points
            
            # 处理前雷达扫描
            self.process_scan(merged, self.last_scan['front'], stamp)
            
            # 处理后雷达扫描（后方）
            self.process_scan(merged, self.last_scan['rear'], stamp, rotation_offset=math.pi)
            
            # 插值填充空隙
            self.fill_gaps(merged)
            
            # 发布融合后的扫描
            self.scan_pub.publish(merged)
            
        except Exception as e:
            rospy.logerr("融合扫描出错: %s", str(e))
    
    def process_scan(self, merged, source_scan, stamp, rotation_offset=0):
        try:
            # 获取坐标系变换
            transform = self.tf_buffer.lookup_transform(
                self.output_frame,
                source_scan.header.frame_id,
                rospy.Time(0),  # 使用最新变换
                rospy.Duration(0.1)
            )
            
            # 处理每个距离值
            for i in range(len(source_scan.ranges)):
                angle = source_scan.angle_min + i * source_scan.angle_increment + rotation_offset
                
                # 计算真实世界中的点位置
                point_distance = source_scan.ranges[i]
                
                # 计算在融合扫描中的索引位置
                idx = self.get_index_in_merged(merged, angle)
                
                if 0 <= idx < len(merged.ranges):
                    # 使用最近的有效值（可改为平均值、最小值等策略）
                    if np.isnan(merged.ranges[idx]) or point_distance < merged.ranges[idx]:
                        merged.ranges[idx] = point_distance
                        if i < len(source_scan.intensities):
                            merged.intensities[idx] = source_scan.intensities[i]
                            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("TF变换异常: %s", str(e))
    
    def get_index_in_merged(self, merged, angle):
        """计算角度在融合扫描中的索引位置"""
        # 确保角度在[-pi, pi]范围内
        normalized_angle = angle
        if normalized_angle > math.pi:
            normalized_angle -= 2 * math.pi
        elif normalized_angle < -math.pi:
            normalized_angle += 2 * math.pi
        
        # 计算索引
        idx = int((normalized_angle - merged.angle_min) / merged.angle_increment)
        return min(max(idx, 0), len(merged.ranges)-1)
    
    def fill_gaps(self, merged):
        """填补扫描中的间隙，创建连续覆盖"""
        max_gap = 15  # 最大间隙点数
        angle_range = merged.angle_max - merged.angle_min
        step = merged.angle_increment
        
        # 扫描填充空隙
        for i in range(len(merged.ranges)):
            # 如果是无效值，尝试填充
            if np.isnan(merged.ranges[i]):
                # 寻找前后有效点
                prev_valid = self.find_nearest_valid(merged, i, -1, max_gap)
                next_valid = self.find_nearest_valid(merged, i, 1, max_gap)
                
                # 如果两边都有有效值，插值填充
                if prev_valid is not None and next_valid is not None:
                    dist_to_prev = abs(i - prev_valid[0])
                    dist_to_next = abs(i - next_valid[0])
                    total_dist = dist_to_prev + dist_to_next
                    
                    # 插值距离
                    w_prev = dist_to_next / total_dist
                    w_next = dist_to_prev / total_dist
                    interp_range = w_prev * prev_valid[1] + w_next * next_valid[1]
                    
                    # 插值强度
                    interp_intensity = w_prev * prev_valid[2] + w_next * next_valid[2]
                    
                    merged.ranges[i] = min(max(interp_range, merged.range_min), merged.range_max)
                    merged.intensities[i] = interp_intensity
    
    def find_nearest_valid(self, scan, start, direction, max_search):
        """在给定方向上查找最近的有效点"""
        for j in range(1, max_search+1):
            idx = start + j * direction
            # 确保在范围内
            if idx < 0 or idx >= len(scan.ranges):
                return None
            # 找到有效点
            if not np.isnan(scan.ranges[idx]):
                return (idx, scan.ranges[idx], scan.intensities[idx])
        return None

if __name__ == '__main__':
    try:
        merger = LaserMerger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass