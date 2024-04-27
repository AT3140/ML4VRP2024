'''
dict_keys(['customer_1', 'customer_10', 'customer_100', 'customer_101', 'customer_102', 'customer_103', 'customer_104', 'customer_105', 'customer_11', 'customer_12', 'customer_13', 'customer_14', 'customer_15', 'customer_16', 'customer_17', 'customer_18', 'customer_19', 'customer_2', 'customer_20', 'customer_21', 'customer_22', 'customer_23', 'customer_24', 'customer_25', 'customer_26', 'customer_27', 'customer_28', 'customer_29', 'customer_3', 'customer_30', 'customer_31', 'customer_32', 'customer_33', 'customer_34', 'customer_35', 'customer_36', 'customer_37', 'customer_38', 'customer_39', 'customer_4', 'customer_40', 'customer_41', 'customer_42', 'customer_43', 'customer_44', 'customer_45', 'customer_46', 'customer_47', 'customer_48', 'customer_49', 'customer_5', 'customer_50', 'customer_51', 'customer_52', 'customer_53', 'customer_54', 'customer_55', 'customer_56', 'customer_57', 'customer_58', 'customer_59', 'customer_6', 'customer_60', 'customer_61', 'customer_62', 'customer_63', 'customer_64', 'customer_65', 'customer_66', 'customer_67', 'customer_68', 'customer_69', 'customer_7', 'customer_70', 'customer_71', 'customer_72', 'customer_73', 'customer_74', 'customer_75', 'customer_76', 'customer_77', 'customer_78', 'customer_79', 'customer_8', 'customer_80', 'customer_81', 'customer_82', 'customer_83', 'customer_84', 'customer_85', 'customer_86', 'customer_87', 'customer_88', 'customer_89', 'customer_9', 'customer_90', 'customer_91', 'customer_92', 'customer_93', 'customer_94', 'customer_95', 'customer_96', 'customer_97', 'customer_98', 'customer_99', 
           'depart', 'distance_matrix', 'instance_name', 'max_vehicle_number', 'vehicle_capacity'])
'''

from vrp_evaluator.utils import load_instance, calculate_distance
import os
from src import helper, util

# load the test case
test_case = 'X-n101-k25'
file = f"/Instances/CVRP/{test_case}.json"
file = os.path.join(os.getcwd(), 'Instances', 'CVRP', 'json', f"{test_case}.json")
inst = load_instance(file)
helper.helper(inst)