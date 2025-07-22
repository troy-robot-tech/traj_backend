import bpy
import numpy as np
import bmesh
import math
import mathutils
import json
import os
from scipy import interpolate, spatial


class POINT():
    def __init__(self, location, normal):
        if not isinstance(location, np.ndarray):
            location = np.array(location)
        if not isinstance(normal, np.ndarray):
            normal = np.array(normal)

        self.location = location
        self.x = location[0]
        self.y = location[1]
        self.z = location[2]

        self.normal = normal

    

class TRAJ():
    def __init__(self, points=[]):
        self.points = []
        self.points = points



    def addPoint(self, point, start=False):
        if start:
            self.points.insert(0, point)
        else:
            self.points.append(point)
        # self.points_sorted_by_x = self.sortbyX(self.points)
    
    def reverse(self):
        self.points = self.points[::-1]

    def sortbyX(self, points):
        return points.sort(key=lambda point: point.x)
    
    def location2point(self, location):
        for point in self.points:
            if abs(point.x - location[0]) < 0.0000001 and abs(point.y - location[1]) < 0.0000001 and abs(point.z - location[2]) < 0.0000001:
                return point
        raise Exception("no points found")
    
    def location2normal(self, location):
        return self.location2point(location).normal
    
    def sort_points_by_direction(self, direction_vector):
        if len(self.points) == 0:
            raise Exception("points is empty")
        
        """
        按照给定方向向量对点进行排序。

        参数:
        - points: list, 坐标点列表。
        - direction_vector: tuple, 方向向量。

        返回:
        - sorted_points: list, 排序后的坐标点列表。
        """
        # 计算每个点在方向向量上的投影长度
        
        self.points.sort(key=lambda point: projection_length(point.location, direction_vector))

    def get_location_list(self, scale=1):
        return [point.location * scale for point in self.points]

    def set_location_list(self, location_list, scale=1):
        if len(location_list) != len(self.points):
            print("location_list length is not equal to points length, redefine all points")
            # raise Warning("location_list length is not equal to points length, redefine all points")
            self.points = []
            for a in location_list:
                point = POINT(a, [0, 0, 0])
                self.addPoint(point)
        else:
            for i, point in enumerate(self.points):
                point.location = location_list[i] * scale
        
    def get_normal_list(self):
        return [point.normal for point in self.points]
    
    def set_normal_list(self, normal_list):
        if len(normal_list) != len(self.points):
            raise Exception("normal_list length is not equal to points length")
        for i, point in enumerate(self.points):
            point.normal = normal_list[i]

    def sparse_points_with_head_tail(self, step, ave=False, num=10):
        res_list = sparse_points_with_head_tail(self.get_location_list(), step, True, num)
        new = []
        for res in res_list:
            new.append(self.location2point(res))
        self.points = new

    def extend_traj(self, traj):
        self.points += traj.points

    def downsample(self, step, keep_node=True, resample=True):
        """
        对轨迹点进行降采样
        :param step: 降采样的步长
        :param keep_node: 是否保留首尾节点
        """
        new_point_location = []
        if step >= len(self.points):
            print('降采样step大于原有点数，不进行降采样')
            return
        
        for i in range(0, len(self.points), step):
            new_point_location.append(self.points[i].location)

        if keep_node:
            new_point_location = new_point_location + [self.points[-1].location]

        if resample:
            new_point_location = resample_curve(new_point_location)
        
        
        
        self.set_location_list(new_point_location[1:-1])



def euler_to_normal(euler, euler_type='zyx'):
    x_euler = euler[0]
    y_euler = euler[1]
    z_euler = euler[2]
    # 初始向量 [0, 0, 1]
    
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
               [0, np.cos(x_euler), -np.sin(x_euler)],
               [0, np.sin(x_euler), np.cos(x_euler)]])

    # 绕y轴旋转的矩阵
    Ry = np.array([[np.cos(y_euler), 0, np.sin(y_euler)],
               [0, 1, 0],
               [-np.sin(y_euler), 0, np.cos(y_euler)]])

    Rz = np.array([[np.cos(z_euler), -np.sin(z_euler), 0],
                [np.sin(z_euler), np.cos(z_euler), 0],
                [0, 0, 1]])
    
    if euler_type.lower() == 'zyx':
        R = np.dot(np.dot(Rz, Ry), Rx)
    else:
        R = np.dot(np.dot(Rx, Ry), Rz)
    res = np.dot(R, np.transpose([0, 0, 1]))
    return list(res)

# 计算平面与三角形的交点（弃用）
def intersect_plane_triangle(plane_origin, plane_normal, v1, v2, v3):
    # 平面的点和法向量
    p0 = plane_origin
    n = plane_normal

    # 三角形的三个顶点
    triangle = [v1, v2, v3]

    # 使用mathutils库的函数进行平面与三角形的交点计算
    intersections = []
    for i in range(3):
        v1 = triangle[i]
        v2 = triangle[(i + 1) % 3]

        # 计算交点
        result = mathutils.geometry.intersect_line_plane(v1, v2, p0, n)
        result = np.array(result)
        if result and is_point_on_segment(result,v1,v2):
            intersections.append(result)
            #print(result)

    return intersections

# 计算平面与多边形的交点
def intersect_plane_duo(plane_origin, plane_normal, verts_set):
    """
    计算平面与某一个多边形的交点。

    参数:
    - plane_origin:  Vector，表示平面上某一点。
    - plane_normal:  Vector，表示平面法向向量。
    - verts_set:  list，多边形的交点列表。

    返回:
    - intersections: list,交点坐标点列表。
    """  
    p0 = plane_origin
    n = plane_normal
    #verts_set.reverse()
    duo = len(verts_set)
    #print(duo)
    # 使用mathutils库的函数进行平面与三角形的交点计算
    intersections = []
    for i in range(duo):
        v1 = verts_set[i]
        v2 = verts_set[(i + 1) % duo]

        # 计算交点
        result = mathutils.geometry.intersect_line_plane(v1, v2, p0, n)
        # print('result', result)
        result_vec = np.array(result)
        # print(type(result_vec))
        try:
            if is_point_on_segment(result_vec, v1, v2):
                intersections.append(result)
        except:
            continue

    return intersections

#判断某个点是否在某条线段上
def is_point_on_segment(P, A, B):
    """
    判断点A是否在线段AB上。

    参数:
    - P: Vector,需要判断的某点坐标。
    - A: Vector,线段AB的A端点坐标。
    - B: Vector,线段AB的B端点坐标。

    返回:
    - T/F: Boolean,在/不在线段上。
    """  
    # 计算向量AB和AP的叉积，判断是否共线
    AB = B - A
    AP = P - A
    cross_product = AB.cross(AP)

    # 如果叉积为零，说明共线
    if cross_product.length < 1e-6:  # 防止浮动误差
        return False

    # 检查点P是否在A和B之间
    if min(A[0], B[0]) <= P[0] <= max(A[0], B[0]) and \
       min(A[1], B[1]) <= P[1] <= max(A[1], B[1]) and \
       min(A[2], B[2]) <= P[2] <= max(A[2], B[2]):
        return True
    return False

def nearest_neighbor_sort(points):
    remaining_points = list(range(len(points)))  # 记录所有点的索引
    sorted_points = [remaining_points.pop(0)]  # 从第一个点开始
    while remaining_points:
        last_point = sorted_points[-1]
        next_point = min(remaining_points, key=lambda point: distance(points[last_point], points[point]))
        sorted_points.append(next_point)
        remaining_points.remove(next_point)
    return [points[i] for i in sorted_points]

# 计算两点之间的欧几里得距离
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

import numpy as np

def compute_euler_angles(v, return_degrees=True):
    """
    计算将 [0, 0, 1] 旋转至目标向量 v 的欧拉角（Z-Y-X 顺序）。
    
    参数:
        v: 目标向量，形如 [x, y, z] 的 NumPy 数组或列表
        return_degrees: 是否返回角度值（默认True），否则返回弧度
    
    返回:
        euler_angles: 欧拉角 (alpha, beta, gamma)，默认单位为度
    """
    v = np.asarray(v, dtype=np.float64)
    x, y, z = v
    
    # 1. 先绕 Y 轴旋转 beta，使向量在 XZ 平面
    beta = np.arctan2(x, z)  # 计算 Y 轴旋转
    
    # 2. 再绕 X 轴旋转 gamma，使向量对齐目标方向
    gamma = -np.arctan2(y, np.sqrt(x**2 + z**2))
    
    # 3. 绕 Z 轴的旋转 alpha 不影响 [0, 0, 1]，设为 0
    alpha = 0.0
    
    euler_angles_rad = np.array([alpha, beta, gamma])
    
    if return_degrees:
        return np.rad2deg(euler_angles_rad)
    else:
        return euler_angles_rad



def projection_length(point, direction_vector):
    """
    计算点在给定方向向量上的投影长度。
    """
    return sum(p * d for p, d in zip(point, direction_vector))

def sort_points_by_direction(points, direction_vector):
    """
    按照给定方向向量对点进行排序。

    参数:
    - points: list, 坐标点列表。
    - direction_vector: tuple, 方向向量。

    返回:
    - sorted_points: list, 排序后的坐标点列表。
    """
    # 计算每个点在方向向量上的投影长度
    projections = [(point, projection_length(point, direction_vector)) for point in points]
    # 按照投影长度进行排序
    projections.sort(key=lambda x: x[1])
    # 提取排序后的点
    sorted_points = [point for point, _ in projections]
    return sorted_points


def sort_points_by_distance(points):
    """
    给点坐标点列表，进行排序。

    参数:
    - points: list, 坐标点列表。

    返回:
    - sorted_points: list, 排序后的坐标点列表。
    """
    sorted_points = [points[0]]  # 选择第一个点作为起始点
    remaining_points = points[1:]  # 其余的点

    while remaining_points:
        last_point = sorted_points[-1]  # 当前已排序的最后一个点
        # 计算当前点到每个剩余点的距离
        distances = [distance(last_point, point) for point in remaining_points]
        # 找到距离最近的点
        closest_point_index = np.argmin(distances)
        closest_point = remaining_points.pop(closest_point_index)

        # 插入排序：找到插入位置
        insert_index = len(sorted_points)
        for i in range(len(sorted_points) - 1, -1, -1):
            if distance(sorted_points[i], closest_point) > distance(sorted_points[i], last_point):
                insert_index = i + 1
                break

        # 插入到排序列表中
        sorted_points.insert(insert_index, closest_point)

    return sorted_points


#生成平面和曲面的交线
def intersection_plane_surface(plane_origin, plane_normal, obj):
    """
    给定平面和曲面，生成两个面的交线坐标列表。

    参数:
    - obj: obj对象。
    - plane_origin:  Vector，表示平面上某一点。
    - plane_normal:  Vector，表示平面法向向量。

    返回:
    - None
    """   
    # 确保物体是网格对象
    if obj.type != 'MESH':
        print("请选中一个网格对象！")
        return None

    # 获取物体的数据
    mesh = obj.data

    # 创建一个BMesh对象，用于操作网格
    bm = bmesh.new()
    bm.from_mesh(mesh)


    # 获取物体中的所有面
    faces = [f for f in bm.faces]

    # 定义平面（可以通过一个点和法向量定义）
    # plane_origin = mathutils.Vector((0, 0, 1010))  # 平面上的一个点
    # plane_normal = mathutils.Vector((0, 0, 1))  # 平面的法向量（假设平面是XY平面）

    # 计算交点
    intersection_points = []

    # 遍历网格的所有面（假设每个面是三角形）
    for face in faces:
        count_verts=len(face.verts)

        #这边尝试计算面的法向量
        face_normal = face.normal  # 获取面的法向量

        verts_set=[v.co for v in face.verts]
        verts_set.reverse()
        points = intersect_plane_duo(plane_origin, plane_normal, verts_set)
        if points:
            intersection_points.extend(points)
        # if len(face.verts) == 3:  # 确保面是三角形
        #     v1, v2, v3 = [v.co for v in face.verts]
        #     # 计算平面和三角形的交点
        #     points = intersect_plane_triangle(plane_origin, plane_normal, v1, v2, v3)
        #     if points:
        #         intersection_points.extend(points)

    return intersection_points

#生成平面和曲面的交线,顺便返回交线每个点对应的法向量捏
def intersection_plane_surface_new(plane_origin, plane_normal, obj):
    """
    给定平面和曲面，生成两个面的交线坐标列表。

    参数:
    - obj: obj对象。
    - plane_origin:  Vector，表示平面上某一点。
    - plane_normal:  Vector，表示平面法向向量。

    返回:
    - None
    """   
    # 确保物体是网格对象
    obj = bpy.data.objects['model'] # todo: change to name
    if obj.type != 'MESH':
        print("请选中一个网格对象！")
        return None

    # 获取物体的数据
    mesh = obj.data

    # 创建一个BMesh对象，用于操作网格
    bm = bmesh.new()
    bm.from_mesh(mesh)


    # 获取物体中的所有面
    faces = [f for f in bm.faces]

    # 定义平面（可以通过一个点和法向量定义）
    # plane_origin = mathutils.Vector((0, 0, 1010))  # 平面上的一个点
    # plane_normal = mathutils.Vector((0, 0, 1))  # 平面的法向量（假设平面是XY平面）

    # 计算交点
    intersection_points = []
    intersection_normals = []

    # 遍历网格的所有面（假设每个面是三角形）
    for face in faces:
        count_verts=len(face.verts)

        #这边尝试计算面的法向量
        face_normal = face.normal  # 获取面的法向量
        # print(face_normal,type(face_normal),len(face_normal),"cccccc")

        verts_set=[v.co for v in face.verts]
        verts_set.reverse()
        points = intersect_plane_duo(plane_origin, plane_normal, verts_set)
        # print(type(points),type(points[0]))
        if points:
            intersection_points.extend(points)
            for i in range(len(points)):
                intersection_normals.append(-face_normal)
        # if len(face.verts) == 3:  # 确保面是三角形
        #     v1, v2, v3 = [v.co for v in face.verts]
        #     # 计算平面和三角形的交点
        #     points = intersect_plane_triangle(plane_origin, plane_normal, v1, v2, v3)
        #     if points:
        #         intersection_points.extend(points)
    print(len(intersection_points),"hhhhhhhhhhh",len(intersection_normals))

    #同时返回交点和法向量
    return intersection_points,intersection_normals


#根据交点表生成相应曲线
def generate_curve(curve_data, intersection_points):
    """
    根据交点表生成相应曲线。

    参数:
    - curve_data: bpy曲线对象。
    - intersection_points: list，交点列表（可以无序）。

    返回:
    - None
    """   
    
    # #排序点
    # intersection_points=sort_points_by_distance(intersection_points)
    #intersection_points=nearest_neighbor_sort(intersection_points)
    # 使用多段曲线生成交线
    polyline = curve_data.splines.new('POLY')
    polyline.points.add(count=len(intersection_points) - 1)
    
    for i, point in enumerate(intersection_points):
        polyline.points[i].co = (point.x, point.y, point.z, 1)  # 添加x, y, z坐标

    # return curve_data

#计算物体在某方向上的跨度
def Calculate_span(obj, plane_normal):
    """
    todo 转变为欧拉角决定
    计算一个点沿某个方向前进一定距离后的新位置。

    参数:
    - obj: obj对象。
    - plane_normal: Vector，表示平面法向向量。

    返回:
    - 当前向量方向上obj对象占据的最大跨度 (float)。
    """
    # 将方向向量归一化（确保只是方向）
    direction_vector = np.array(plane_normal)
    direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)

    # 计算每个点在该方向上的投影值
    projections = []
    
    # 确保物体是网格对象
    if obj.type != 'MESH':
        print("请选中一个网格对象！")
        return None

    # 获取物体的数据
    mesh = obj.data

    # 创建一个BMesh对象，用于操作网格
    bm = bmesh.new()
    bm.from_mesh(mesh)


    # 获取物体中的所有面
    faces = [f for f in bm.faces]
    verts_set=[]
    # 遍历网格的所有面（假设每个面是三角形）
    for face in faces:
        for v in face.verts:
            verts_set.append(v)
            point=np.array([v.co.x, v.co.y, v.co.z])
            projection = np.dot(point, direction_unit_vector)
            projections.append(projection)
    
    # 找到最负和最正的投影值以及对应的点
    min_projection_index = np.argmin(projections)  # 最负的点索引
    max_projection_index = np.argmax(projections)  # 最正的点索引

    span={}

    # 获取最负点和最正点的投影值（即在该方向上的距离）
    min_distance = projections[min_projection_index]
    max_distance = projections[max_projection_index]

    span["min_vert"] = np.array([verts_set[min_projection_index].co.x,verts_set[min_projection_index].co.y,verts_set[min_projection_index].co.z])  # 最负的点
    span["max_vert"] = np.array([verts_set[max_projection_index].co.x,verts_set[max_projection_index].co.y,verts_set[max_projection_index].co.z])  # 最正的点
    span["distance"] = max_distance-min_distance

    return span

#手动计算物体在某方向上的跨度
def Calculate_span_manual(max_vert,min_vert, plane_normal):
    """
    。

    参数:
    - max_vert: 手动选取最大点。
    - min_vert: 手动选取最小点。
    - plane_normal: Vector，表示平面法向向量。

    返回:
    -  (float)。
    """
    # 将方向向量归一化（确保只是方向）
    direction_vector = np.array(plane_normal)
    direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)

    max_point=np.array([max_vert.x,max_vert.y,max_vert.z])
    max_projection = np.dot(max_point, direction_unit_vector)
    min_point=np.array([min_vert.x,min_vert.y,min_vert.z])
    min_projection = np.dot(min_point, direction_unit_vector)

    if max_projection<min_projection:
        max_point,min_point=min_point,max_point
        max_projection,min_projection=min_projection,max_projection

    span={}

    span["min_vert"] = min_point# 最负的点
    span["max_vert"] = max_point  # 最正的点
    span["distance"] = max_projection-min_projection

    return span

def move_point_along_direction(point, direction_vector, distance):
    """
    计算一个点沿某个方向前进一定距离后的新位置。

    参数:
    - point: numpy 向量，表示原始点 (x, y, z)。
    - direction_vector: numpy Vector向量，表示方向向量 (dx, dy, dz)。
    - distance: 前进的距离 (float)。

    返回:
    - 新的位置 (numpy 向量)。
    """
    # 归一化方向向量
    unit_vector = direction_vector / np.linalg.norm(direction_vector)
    
    # 计算新的位置
    new_position = point + distance * unit_vector
    
    return new_position


        
#常规计算切割平面
#plane_normal即可确定沿着哪个坐标轴进行移动切割
def calculate_cutting_plane(obj, plane_normal, overlap_spacing):
    """
    计算对于当前模型，以给定法向量的平面，一定的叠枪距离，计算出一组切割平面。

    参数:
    - obj: obj对象。
    - plane_normal: Vector，表示平面法向向量。
    - overlap_spacing: 叠枪距离 (int)。

    返回:
    - 一组平面的原点 (list)。
    """
    bias=overlap_spacing / 6.5
    obj_span = Calculate_span(obj, plane_normal)
    #这里其实不能简单用distance/overlap_spacing来计算切割平面的数量，因为方向可能不一致，所以当切割平面角度特定情况时，数组会超限
    #后续再更改debug吧，先记录下来
    cutting_num = math.floor(obj_span["distance"] / overlap_spacing)
    #这里其实是相当于手动确定要不要给切割平面再加一个面，因为有时候为了覆盖更全面，还需要再来一个面
    if(bias > overlap_spacing/4):
        cutting_num += 1
    plane_set = []
    point = obj_span["min_vert"]
    #使得切割平面起点在最小点的前面
    #右后翼子板用-overlap_spacing/2
    point = move_point_along_direction(point, plane_normal, -bias)
    for i in range(cutting_num):
        plane = {}
        plane["plane_normal"] = plane_normal
        plane["plane_origin"] = move_point_along_direction(point, plane_normal, overlap_spacing) 
        point=plane["plane_origin"]
        plane_set.append(plane)
        
        #print(point)

    return plane_set

#手动计算切割平面
#plane_normal即可确定沿着哪个坐标轴进行移动切割
def calculate_cutting_plane_manual(manual_span, plane_normal, overlap_spacing, bias,cutting_num_bias):
    """
    计算对于当前模型，以给定法向量的平面，一定的叠枪距离，计算出一组切割平面。

    参数:
    - point_normal_set: 列表，存储一组字典，每个字典包含（1）手动选取的两个点，（2）两个点进行切割的平面法向量。
    - overlap_spacing: 叠枪距离 (int)。

    返回:
    - 一组平面的原点 (list)。
    """
    bias=bias
    #这里obj_span是手动设置的，相当于虚构了一个物体对象
    obj_span=manual_span
    cutting_num=math.floor(obj_span["distance"]/overlap_spacing)
    #这里其实是相当于手动确定要不要给切割平面再加一个面，因为有时候为了覆盖更全面，还需要再来一个面
    # if(bias > overlap_spacing/4):
    #     cutting_num+=1
    cutting_num+=cutting_num_bias
    plane_set=[]
    point=obj_span["min_vert"]
    #使得切割平面起点在最小点的前面
    #右后翼子板用-overlap_spacing/2
    point=move_point_along_direction(point, plane_normal, -bias)
    for i in range(cutting_num):
        plane={}
        plane["plane_normal"]=plane_normal
        plane["plane_origin"]=move_point_along_direction(point, plane_normal, overlap_spacing) 
        point=plane["plane_origin"]
        plane_set.append(plane)
        
        #print(point)

    return plane_set


def extend_curve(points, extend_length, at_start=True):
    """
    延伸曲线的头部或尾部。

    参数:
    - points: list, 曲线的坐标点列表 (Vector 格式)。
    - extend_length: float, 延伸的长度。
    - at_start: bool, 如果为True，则在头部延伸，否则在尾部延伸。

    返回:
    - extended_points: list, 延伸后的坐标点列表 (Vector 格式)。
    """

    if at_start:
        # 计算头部方向向量
        # 初始是使用point[0]和point[1]的向量，但是这样会由于模型在一些边角地方弯曲导致延伸的方向不对，所以改成使用point[1]和point[2]的向量
        direction_vector = np.array(points[0]) - np.array(points[1])
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        # 计算新的起点
        new_start_point = np.array(points[0]) + direction_vector * extend_length
        # 插入新的起点
        extended_points = [np.array(new_start_point.tolist())] + points
    else:
        # 计算尾部方向向量
        # 初始是使用point[-1]和point[-2]的向量，但是这样会由于模型在一些边角地方弯曲导致延伸的方向不对，所以改成使用point[-2]和point[-3]的向量
        direction_vector = np.array(points[-1]) - np.array(points[-2])
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        # 计算新的终点
        new_end_point = np.array(points[-1]) + direction_vector * extend_length
        # 添加新的终点
        extended_points = points + [np.array(new_end_point.tolist())]

    return extended_points

def extend_curve_fixed(points, extend_length, loop_num, at_start=True):
    """
    延伸曲线的头部或尾部。

    参数:
    - points: list, 曲线的坐标点列表 (Vector 格式)。
    - extend_length: float, 延伸的长度。
    - at_start: bool, 如果为True，则在头部延伸，否则在尾部延伸。

    返回:
    - extended_points: list, 延伸后的坐标点列表 (Vector 格式)。
    """
    if at_start:
        # 计算头部方向向量
        # 初始是使用point[0]和point[1]的向量，但是这样会由于模型在一些边角地方弯曲导致延伸的方向不对，所以改成使用point[1]和point[2]的向量
        direction_vector = np.array(points[2]) - np.array(points[3])
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        for i in range(loop_num):
            # 计算新的起点
            new_start_point = np.array(points[0]) + direction_vector * extend_length
            # 插入新的起点
            extended_points = [np.array(new_start_point.tolist())] + points
            points=extended_points
    else:
        # 计算尾部方向向量
        # 初始是使用point[-1]和point[-2]的向量，但是这样会由于模型在一些边角地方弯曲导致延伸的方向不对，所以改成使用point[-2]和point[-3]的向量
        direction_vector = np.array(points[-2]) - np.array(points[-4])
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        for i in range(loop_num):
            # 计算新的终点
            new_end_point = np.array(points[-1]) + direction_vector * extend_length
            # 添加新的终点
            extended_points = points + [np.array(new_end_point.tolist())]
            points=extended_points

    return extended_points

# 新增映射拉出向量的延长函数

def extend_curve_fixed_map_new(traj: TRAJ, extend_length, loop_num, at_start=True):
    """
    延伸曲线的头部或尾部。

    参数:
    - points: list, 曲线的坐标点列表 (Vector 格式)。
    - extend_length: float, 延伸的长度。
    - loop_num: int, 采样的点数目
    - at_start: bool, 如果为True，则在头部延伸，否则在尾部延伸。

    返回:
    - extended_points: list, 延伸后的坐标点列表 (Vector 格式)。
    """
    # print('points number ', len(points))
    # print('extend_map', extend_map)
    if len(traj.points) < 2:
        return traj
    
    if at_start:
        # 计算头部方向向量
        # 初始是使用point[0]和point[1]的向量，但是这样会由于模型在一些边角地方弯曲导致延伸的方向不对，所以改成使用point[1]和point[2]的向量
        try:
            direction_vector = np.array(traj.points[2].location) - np.array(traj.points[3].location)
        except:
            direction_vector = np.array(traj.points[0].location) - np.array(traj.points[1].location)
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        
        if len(traj.points) == 2:
            mean_vector = traj.points[0].normal
        elif len(traj.points) == 2:
            mean_vector=(traj.points[0].normal + traj.points[1].normal)/2
        else:
            mean_vector=(traj.points[0].normal + traj.points[1].normal + traj.points[2].normal)/3

        for i in range(loop_num):
            # 计算新的起点
            new_start_location = traj.points[0].location + (direction_vector * extend_length)

            # 插入点也需要映射拉出位置
            # new_start_point_vector=mathutils.Vector(new_start_point.tolist())
            new_start_point = POINT(new_start_location, mean_vector)
    
            # 插入新的起点
            traj.addPoint(new_start_point, True)
  
    else:
        # 计算尾部方向向量
        # 初始是使用point[-1]和point[-2]的向量，但是这样会由于模型在一些边角地方弯曲导致延伸的方向不对，所以改成使用point[-2]和point[-3]的向量
        try:
            direction_vector = traj.points[-2].location - traj.points[-4].location
        except:
            direction_vector = traj.points[-1].location - traj.points[-2].location

        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        if len(traj.points) == 1:
            mean_vector = traj.points[-1].normal
        elif len(traj.points) == 2:
            mean_vector = (traj.points[-1].normal + traj.points[-2].normal) / 2
        else:
            mean_vector = (traj.points[-1].normal + traj.points[-2].normal + traj.points[-3].normal) / 3

        for i in range(loop_num):
            # 计算新的终点
            new_end_location = traj.points[-1].location + direction_vector * extend_length

            # 插入点也需要映射拉出位置
            new_start_point = POINT(new_end_location, mean_vector)

            # 添加新的终点
            traj.addPoint(new_start_point, False)


    return traj

def sparse_points_with_head_tail(points, step, ave=False, num=10):
    # print('points', points)
    """
    稀疏坐标点列表，但保留头尾。

    参数:
    - points: list, 坐标点列表。
    - step: int, 每隔多少个点保留一个点。
    - ave: bool 是否对稀疏后的点平均化
    - num: int, 若需要平均化，则采样的点数

    返回:
    - sparse_points: list, 稀疏后的坐标点列表。
    """
    if len(points) <= 2:
        return points  # 如果点数少于等于2，直接返回
    # elif len(points) <= 10 * step:
    #     return points[::step-5]

    # 保留头尾，中间部分进行稀疏
    

    if ave:
        m = len(points)
        print('m', m)
        # 计算中间采样点数
        num_samples = (m - 1) // num
        # 初始化采样结果，包含头尾两个点
        sparse_points = []

        # 从第1个元素开始，每隔n个元素取一个样本
        for i in range(1, num_samples + 1):
            index = i * num
            if index != 0 and index != m - 1:  # 排除头尾两个点
                sparse_points.append(points[index])
        # sparse_points.append(points[-1])

    else:
        sparse_points = [points[0]] + points[step * 2:-step - 4:step] + [points[-1]]

    return sparse_points

def split_list_by_indices(traj, indices):
    """
    根据索引列表将坐标点列表切分成多个子列表。

    参数:
    - points: list, 坐标点列表。
    - indices: list, 索引列表。

    返回:
    - split_points: list, 切分后的子列表。
    """
    split_list = []
    split_points = []
    start = 0
    
    for index in indices:
        split_points.append(traj.points[start:index + 1])
        start = index + 1

    split_points.append(traj.points[start:])  # 添加最后一个子列表
    for points in split_points:
        split_list.append(TRAJ(points))
    return split_list

def fix_curve(points):
    """
    修正曲线，使得曲线的点之间的距离不会过于接近，
    再就是如果是前后引擎盖出现中间空缺的情况，这时候就需要在轨迹中间做一些延长

    参数:
    - points: list, 坐标点列表(已经排序好)。

    返回:
    - fixed_points: list, 修正后的坐标点列表。
    """

    hole_index=[]
    fix_curve_points=[]

    if(len(points)<20):
        return points

    for i in range(len(points)-1):
        if i==0:
            continue
        # #这里设置为10会出错，暂未排查原因，先设置为8，左前翼子板设置为6
        # if distance(points[i],points[i-1])<5:
        #     points.pop(i)
        #     continue

    for i in range(len(points)-1):
        if i==0:
            continue
        # #这里设置为10会出错，暂未排查原因，先设置为8
        # if distance(points[i],points[i-1])<8:
        #     points.pop(i)
        #     i-=1
        #     continue
        elif distance(points[i],points[i-1])>200:
            hole_index.append(i-1)
            continue
    
    sub_points_set = split_list_by_indices(points,hole_index)
    for i in range(len(hole_index)):
        sub_points_set[i]=extend_curve_fixed(sub_points_set[i], 30,4, at_start=False)
        sub_points_set[i+1]=extend_curve_fixed(sub_points_set[i+1], 30,4, at_start=True)
        # for j in range(4):
        #     sub_points_set[i]=extend_curve(sub_points_set[i], 30, at_start=False)
        #     sub_points_set[i+1]=extend_curve(sub_points_set[i+1], 30, at_start=True)
    
    for sub_points in sub_points_set:
        fix_curve_points.extend(sub_points)
    
    return fix_curve_points

def fix_curve_map(points, extend_map):
    """
    修正曲线，使得曲线的点之间的距离不会过于接近，
    再就是如果是前后引擎盖出现中间空缺的情况，这时候就需要在轨迹中间做一些延长

    参数:
    - points: list, 坐标点列表(已经排序好)。

    返回:
    - fixed_points: list, 修正后的坐标点列表。
    """

    hole_index = []
    fix_curve_points = []

    if(len(points) < 20):
        return points

    for i in range(len(points) - 1):
        if i==0:
            continue
        # #这里设置为10会出错，暂未排查原因，先设置为8，左前翼子板设置为6
        # if distance(points[i],points[i-1])<5:
        #     points.pop(i)
        #     continue

    for i in range(len(points) - 1):
        if i == 0:
            continue
        # #这里设置为10会出错，暂未排查原因，先设置为8
        # if distance(points[i],points[i-1])<8:
        #     points.pop(i)
        #     i-=1
        #     continue
        elif distance(points[i], points[i - 1])>200:
            hole_index.append(i - 1)
            continue
    
    sub_points_set=split_list_by_indices(points, hole_index)
    for i in range(len(hole_index)):
        sub_points_set[i] = extend_curve_fixed_map_new(sub_points_set[i], 30, 4, extend_map, at_start=False)
        sub_points_set[i + 1] = extend_curve_fixed_map_new(sub_points_set[i + 1], 30, 4, extend_map, at_start=True)
        # for j in range(4):
        #     sub_points_set[i]=extend_curve(sub_points_set[i], 30, at_start=False)
        #     sub_points_set[i+1]=extend_curve(sub_points_set[i+1], 30, at_start=True)
    
    for sub_points in sub_points_set:
        fix_curve_points.extend(sub_points)
    
    return fix_curve_points

def fix_curve_map_new(traj: TRAJ):
    """
    修正曲线，使得曲线的点之间的距离不会过于接近，
    再就是如果是前后引擎盖出现中间空缺的情况，这时候就需要在轨迹中间做一些延长

    参数:
    - points: list, 坐标点列表(已经排序好)。

    返回:
    - fixed_points: list, 修正后的坐标点列表。
    """

    hole_index = []

    if(len(traj.points) < 20):
        return traj

    for i in range(len(traj.points) - 1):
        if i == 0:
            continue
        # #这里设置为10会出错，暂未排查原因，先设置为8，左前翼子板设置为6
        # if distance(points[i],points[i-1])<5:
        #     points.pop(i)
        #     continue

    for i in range(len(traj.points) - 1):
        if i == 0:
            continue

        # #这里设置为10会出错，暂未排查原因，先设置为8
        # if distance(points[i],points[i-1])<8:
        #     points.pop(i)
        #     i-=1
        #     continue
        elif np.linalg.norm(traj.points[i].location - traj.points[i - 1].location) > 200:
            hole_index.append(i - 1)
            continue
    
    sub_points_set = split_list_by_indices(traj, hole_index) # list of points
    print('sub_points_set', sub_points_set)
    for i in range(len(hole_index)):
        sub_points_set[i] = extend_curve_fixed_map_new(sub_points_set[i], 30, 4, at_start=False)
        sub_points_set[i + 1] = extend_curve_fixed_map_new(sub_points_set[i + 1], 30, 4, at_start=True)
        # for j in range(4):
        #     sub_points_set[i]=extend_curve(sub_points_set[i], 30, at_start=False)
        #     sub_points_set[i+1]=extend_curve(sub_points_set[i+1], 30, at_start=True)
    
    fix_curve_points = []
    for sub_points in sub_points_set:
        fix_curve_points.extend(sub_points.points)
    
    return TRAJ(fix_curve_points)

def smooth_vectors(vectors, window_size=3):
    """
    对向量列表进行平滑处理。

    参数:
    - vectors: list[mathutils.Vector], 按序排列的向量列表。
    - window_size: int, 滑动窗口的大小（必须为奇数）。

    返回:
    - smoothed_vectors: list[mathutils.Vector], 平滑后的向量列表。
    """
    half_window = window_size // 2
    smoothed_vectors = []

    for i in range(len(vectors)):
        # 获取滑动窗口内的向量
        start = max(0, i - half_window)
        end = min(len(vectors), i + half_window + 1)
        window = vectors[start:end]
        # print(window, 'window')
        # 计算窗口内向量的平均值
        stacked_arrays = np.stack(window)
        mean_vector = np.sum(stacked_arrays, axis=0) / len(window)
        smoothed_vectors.append(mean_vector)

    return smoothed_vectors

def fix_outliers(vectors, threshold=0.5, window_size=3):
    """
    修正向量列表中的异常向量。

    参数:
    - vectors: np.array(locations), 按序排列的向量列表。
    - threshold: float, 异常检测的阈值（与前后向量的差异）。
    - window_size: int, 用于修正的滑动窗口大小。

    返回:
    - fixed_vectors: list[mathutils.Vector], 修正后的向量列表。
    """
    fixed_vectors = vectors[:]
    for i in range(1, len(vectors) - 1):
        # 计算当前向量与前后向量的差异
        diff_prev = np.linalg.norm(vectors[i] - vectors[i - 1])
        diff_next = np.linalg.norm(vectors[i] - vectors[i + 1])

        # 如果差异超过阈值，认为是异常向量
        if diff_prev > threshold or diff_next > threshold:
            # 用滑动窗口内的平均值替代异常向量
            start = max(0, i - window_size // 2)
            end = min(len(vectors), i + window_size // 2 + 1)
            window = vectors[start:end]
            fixed_vectors[i] = sum(window, np.array([0, 0, 0])) / len(window)

    return fixed_vectors

def smooth_and_fix_vectors(vectors, smooth_window=3, fix_threshold=0.5, fix_window=3):
    """
    对向量列表进行平滑处理并修正异常向量。

    参数:
    - vectors: list[mathutils.Vector], 按序排列的向量列表。
    - smooth_window: int, 平滑处理的滑动窗口大小。
    - fix_threshold: float, 异常检测的阈值。
    - fix_window: int, 修正异常向量的滑动窗口大小。

    返回:
    - processed_vectors: list[mathutils.Vector], 处理后的向量列表。
    """
    # 平滑处理
    smoothed_vectors = smooth_vectors(vectors, window_size=smooth_window)

    # 修正异常向量
    fixed_vectors = fix_outliers(smoothed_vectors, threshold=fix_threshold, window_size=fix_window)

    return fixed_vectors

def resample_curve(points):
    """
    使曲线上的点间隔均匀化，同时保持点数目不变
    
    参数:
        points: 原始曲线顶点列表，格式为np.array的列表 [np.array([x1,y1,...]), ...]
    
    返回:
        重采样后的顶点列表，点数与输入相同
    """
    # 将输入点转换为numpy数组
    points_array = np.array([p for p in points])
    dim = points_array.shape[1]  # 获取点的维度(2D, 3D等)
    
    # 固定首尾两个点
    fixed_points = [points_array[0], points_array[-1]]
    
    # 计算累积弦长(各顶点间的距离累计)
    diffs = np.diff(points_array, axis=0)
    dists = np.sqrt((diffs**2).sum(axis=1))  # 计算每两点间的距离
    cum_dists = np.insert(np.cumsum(dists), 0, 0)  # 从0开始的累积距离
    
    # 计算新的均匀间隔的参数值(点数与输入相同，不包括首尾两个点)
    new_params = np.linspace(0, cum_dists[-1], len(points) - 2)
    
    # 为每个维度创建插值函数
    interpolators = []
    for i in range(dim):
        # 使用线性插值保持原始点的精确位置
        interpolator = interpolate.interp1d(
            cum_dists, 
            points_array[:, i], 
            kind='linear',
            fill_value='extrapolate'
        )
        interpolators.append(interpolator)
    
    # 在新的参数点上插值
    new_points = [fixed_points[0]]
    for param in new_params:
        point = []
        for i in range(dim):
            point.append(interpolators[i](param))
        new_points.append(np.array(point))
    new_points.append(fixed_points[1])
    
    return new_points


def paint_plain(location, euler):
    """
    给定法向量与旋转值，创建一个平面
    可视化切割平面
    """
    # 创建一个平面
    bpy.ops.object.empty_add(type='CUBE', location=list(location / 1000), scale=(1, 1, 0))

    # 获取创建的平面对象
    plane = bpy.context.object
    plane.scale = (1, 1, 0)

    # 设置平面的旋转
    plane.rotation_euler = euler
        
def print_traj(traj, index=0):

    mesh = bpy.data.meshes.new(name="TrajLine" + str(index))
    obj = bpy.data.objects.new(name="TrajLineObject"+ str(index), object_data=mesh)

    # 将对象链接到当前场景中
    scene = bpy.context.scene
    scene.collection.objects.link(obj)
    
    # 定义顶点和面
    mesh.from_pydata(traj.get_location_list(0.001), [], [])

def pull_traj(traj, pull_distance, tree):
    """
    沿给定模型，搜索轨迹点与最相近模型点，取该模型点法向，拉出轨迹点
    """
    normal_list = []
    print(traj.points[0].location, 'location')
    for index, point in enumerate(traj.points):
        mesh = obj.data
        # 使用KD树查询最近的顶点
        distance, vertex_index = tree.query(point.location)
        vertex_normal = mesh.vertices[vertex_index].normal
        normal_list.append(np.array(vertex_normal))

    pull_distance = 50
    for i, point in enumerate(traj.points):
        traj.points[i].location = point.location + normal_list[i] * pull_distance
        traj.points[i].normal = -normal_list[i]
    
    return traj

cut_plan=1

#以下相当于main函数，脚本中我直接写过程了就
bpy.ops.object.mode_set(mode='OBJECT')
# 获取当前选中的物体
obj = bpy.context.object

#设置叠枪距离 
#后保险杠用90，引擎盖用80，前翼子板用也是80，后翼子板用100
overlap_spacing = 80
# points_map_normals={}



if cut_plan==1:


    #定义平面（可以通过一个点和法向量定义）
    #todo：后续改成数组，存储一组平面
    # plane_origin = mathutils.Vector((0, 0, 1010))  # 平面上的一个点
    #右后翼子板设置成(-0.1,0,1),引擎盖设置成(1,0,0),左前翼子板设置成(1,0,1)
    #左前翼子板一组调试好的参数是(1,0.04,1)，bias设置成overlap_spacing/6.5
    #左前翼子板手动用平面切割出来了一条比较完美的平面，法向量是(0.5376, 0.0949, 0.8379)
    plane_euler = (0, 0, 0)  # 平面的旋转，弧度制
    plane_normal = euler_to_normal(plane_euler)

    #plane_set在面对复杂曲面（例如前翼子板）的时候其实可以考虑自行设置，这里是自动计算的
    plane_set = calculate_cutting_plane(obj, plane_normal, overlap_spacing)
    file_path = r"D:\轨迹生成\out\plane_set_res_right.json"

    trij_surface_set=[]

    #定义排序方向，引擎盖用（0,1,0）,右后翼子板用（0,-1,0），左前翼子板用（0, -1, 0）
    direction_vector = np.array([0, 1, 0]) # 沿Y轴方向排序坐标点
    print('plane_set', plane_set)
    for plane_dict in plane_set:
        # 可视化切割平面
        paint_plain(
            location=plane_dict["plane_origin"],
            normal=plane_dict["plane_normal"]
        )
        
    # 计算点范围，建立kdtree检索
    obj = bpy.data.objects['model']
    model_vertices_pos = []
    for i in range(len(obj.data.vertices)):
        co = np.array(obj.data.vertices[i].co)
        model_vertices_pos.append(co)
    
    # 拉出轨迹点
    model_target_tree = None
    model_target_tree = spatial.cKDTree(model_vertices_pos)
    print("模型点之间的kdtree完成")

    for index, plane in enumerate(plane_set):
        
        #调用新函数，同时返回交点和法向量
        # 这里的intersection_points是交点，intersection_normals是法向量
        intersection_points, intersection_normals = intersection_plane_surface_new(plane["plane_origin"], plane["plane_normal"], obj)

        #将交点存入traj_points
        traj_points = TRAJ([])
        for i in range(len(intersection_points)):
            point = POINT(intersection_points[i], intersection_normals[i])
            traj_points.addPoint(point)

        # 排序点
        traj_points.sort_points_by_direction(direction_vector)
        traj_points.downsample(15, keep_node=True, resample=True)
        # print_traj(traj_points, index) # ok

        intersection_points = fix_curve_map_new(traj_points)
        print_traj(intersection_points, index) # ok

        #拉出顶点
        traj_points = pull_traj(traj_points, 50, model_target_tree)
        
        intersection_points = extend_curve_fixed_map_new(traj_points, 30, 4, at_start=True)
        intersection_points = extend_curve_fixed_map_new(intersection_points, 30, 4, at_start=False)
        # print_traj(intersection_points, index) # ok
        
        if(index % 2 == 1):
            intersection_points.reverse()
        print('extend_curve_fixed_map_', len(intersection_points.points))
        trij_surface_set.append(intersection_points) # trij_surface_set: list of TRAJ


    # #反转一下，从上往下开始
    trij_surface_set.reverse()
    print(len(trij_surface_set), 'len(trij_surface_set)')
    # 将所有traj合并为一个
    trij_surface_set_flattened = trij_surface_set[0]
    for trij_surface in trij_surface_set:
        trij_surface_set_flattened.extend_traj(trij_surface)
        # print(trij_surface.points[0].location)

    # 总输出
    print_traj(trij_surface_set_flattened, 0)



    

#保存json，传给嘉孙
traj_json_file={}


vertex_coords = [[v[0], v[1], v[2]] for v in trij_surface_set_flattened.get_location_list()]

#这里暂时不用
algo={
    "version": "v0.1.0",
    "dist_gun_overlap": 80.0,
    "dist_interval": 100.0,
    "dist_to_surface": 0,
    "speed": 300.0,
    "accelerate": 0,
    "flow": 100,
    "sector": 50,
    "pressure": 50,
    "gen_edge_traj": False,
    "opt_normal_position": True,
}
traj_json_file["algo"] = algo

process={
    "process_type": "VarnishFog",
    "color": "Light",
    "oil_brand": "BASF",
}
traj_json_file["process"]=process



# 这里后续需要保存成4份json然后分别对应四道工艺
print('trij_surface_set', trij_surface_set_flattened)
index=0
traj_json_file["traj_surface"]=[]
# for i, traj in enumerate(trij_surface_set):
for i, traj in enumerate([trij_surface_set_flattened]):
    for j, point in enumerate(traj.points):
        spray = True
        if(j == 0):
            spray = False
 
        
        traj_surface_item={}
        traj_surface_item["p"]=list(point.location)
        traj_surface_item["n"]=list(point.normal)
        traj_surface_item["speed"] = 350
        traj_surface_item["index"] = index
        index += 1
        traj_surface_item["spray"] = spray
        traj_surface_item["posture"] = traj_surface_item["p"] + list(compute_euler_angles(point.normal, return_degrees=True))
        traj_surface_item["transition"] = False
        traj_surface_item["gun_posture"] = "default"

        traj_json_file["traj_surface"].append(traj_surface_item)

# 获取当前脚本所在目录
script_dir = r"D:\轨迹生成"
# 定义目标文件夹和文件名
output_folder = os.path.join(script_dir, "saved_json")
os.makedirs(output_folder, exist_ok=True)  # 确保文件夹存在，如果不存在则创建
# 定义文件路径




filepath = os.path.join(output_folder, obj.name+"_"+str(overlap_spacing)+'test'+".json")
 # 将数据保存为JSON文件
with open(filepath, 'w') as json_file:
    json.dump(traj_json_file, json_file, indent=4)



