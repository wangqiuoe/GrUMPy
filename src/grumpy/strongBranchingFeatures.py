import numpy as np
from copy import deepcopy
class strongBranchingFeatures(object):

    def __init__(self, CONSTRAINTS, VARIABLES, OBJ, MAT, RHS):
        self.CONSTRAINTS = CONSTRAINTS
        self.VARIABLES = VARIABLES
        self.OBJ = OBJ
        self.MAT = MAT
        self.RHS = RHS
        self.StaticFeaName = {
                1:'ratio_ci_sum_ci',
                3:'max_M_j1_plus',
                4:'min_M_j1_plus',
                5:'max_M_j1_minus',
                6:'min_M_j1_minus',
                7:'max_M_j2',
                8:'min_M_j2',
                11:'max_M_j3',
                12:'min_M_j3',

        }
        self.NodeFeaName = {
            0: 'reduced_cost_normalize_by_ci',
            1: 'fractionality_to_up',
            2: 'fractionality_to_down',
            3: 'proportion_of_fixed_variables'
        }
        self.BrachingFeaName = {
            0: 'max_obj_decrease_ratio_from_parent',
            1: 'min_obj_decrease_ratio_from_parent',
            2: 'mean_obj_decrease_ratio_from_parent',
            3: 'std_obj_decrease_ratio_from_parent',
            4: 'quart1_obj_decrease_ratio_from_parent',
            5: 'quart2_obj_decrease_ratio_from_parent',
            6: 'quart3_obj_decrease_ratio_from_parent',
            7: 'ratio_of_num_branching_i_with_num_branching'
        }
        self.staticFeature = self.getStaticFeature()
        self.obj_decrease_ratio_from_parent = {}
        self.num_branching = {}
        self.AllFeaName = self.getAllFeaName()
        self.fea_col = [
            's_01',
            's_03',
            's_04',
            's_05',
            's_06',
            's_07',
            's_08',
            's_11',
            's_12',
            'n_00',
            'n_01',
            'n_02',
            'n_03',
            'b_00',
            'b_01',
            'b_02',
            'b_03',
            'b_04',
            'b_05',
            'b_06',
            'b_07'
        ]

    def getAllFeaName(self):
        fea_type = {
            's': self.StaticFeaName,
            'n': self.NodeFeaName,
            'b': self.BrachingFeaName,
        }
        fmap_all = {}
        for fea_type, fmap in fea_type.items():
            for k,v in fmap.items():
                key = "%s_%s" %(fea_type, '0'+str(k) if k < 10 else k)
                fmap_all[key] = v
        return fmap_all

    def getStaticFeature(self,):
        fmap_static = {}
        b = self.RHS
        c = self.OBJ
        nConstr = len(b)
        for i in self.VARIABLES:
            fmap = {}
            ci = c[i]
            Ai = self.MAT[i]

            # |ci| / sum of |ci| such that ci >= 0
            sum_ci = sum([abs(v) for v in c.values()])
            fmap[1] = self.get_division(abs(ci),sum_ci)
            
            # M1
            M_j1_plus = [self.get_division(Ai[j],abs(b[j])) for j in range(nConstr) if b[j] >= 0]
            M_j1_minus = [self.get_division(Ai[j],abs(b[j])) for j in range(nConstr) if b[j] < 0]
            fmap[3] = self.get_max(M_j1_plus)
            fmap[4] = self.get_min(M_j1_plus)
            fmap[5] = self.get_max(M_j1_minus)
            fmap[6] = self.get_min(M_j1_minus)
        
            # M2
            M_j2 =  [self.get_division(abs(ci),Ai[j]) for j in range(nConstr) if ci]
            fmap[7] = self.get_max(M_j2)
            fmap[8] = self.get_min(M_j2)

            # M3
            sum_Ajk = []
            for j in range(nConstr):
                sum_Ajk.append(sum([abs(self.MAT[k][j]) for k in self.VARIABLES]))
            M_j3 =  [self.get_division(abs(Ai[j]),sum_Ajk[j]) for j in range(nConstr)]
            # NOTE, miss M_j3_plusplus and M_j3_plusminus, etc., 
            fmap[11] = self.get_max(M_j3)
            fmap[12] = self.get_min(M_j3)

            fmap_static[i] = fmap
        return fmap_static

    def getNodeFeature(self, var_reduced_cost, frac_to_ceil, frac_to_floor, n_fixed_vars):
        fmap_node = {}
        for i in self.VARIABLES:
            fmap = {}
            fmap[0] = var_reduced_cost[i]/abs(self.OBJ[i]+0.1)
            fmap[1] = frac_to_ceil[i]
            fmap[2] = frac_to_floor[i]
            fmap[3] = n_fixed_vars/len(self.VARIABLES)
            fmap_node[i] = fmap
        return fmap_node

    def getBranchingFeature(self, i):
        fmap = {}
        fmap[0] = self.get_max(self.obj_decrease_ratio_from_parent.get(i,[]))
        fmap[1] = self.get_min(self.obj_decrease_ratio_from_parent.get(i,[]))
        fmap[2] = self.get_mean(self.obj_decrease_ratio_from_parent.get(i,[]))
        fmap[3] = self.get_std(self.obj_decrease_ratio_from_parent.get(i,[]))
        fmap[4] = self.get_quartile(self.obj_decrease_ratio_from_parent.get(i,[]), 0.25)
        fmap[5] = self.get_quartile(self.obj_decrease_ratio_from_parent.get(i,[]), 0.5)
        fmap[6] = self.get_quartile(self.obj_decrease_ratio_from_parent.get(i,[]), 0.75)
        if sum(self.num_branching.values()) == 0:
            fmap[7] = 0
        else:
            fmap[7] = self.num_branching.get(i,0)/sum(self.num_branching.values())
        return fmap



    def getFeature(self,i, fmap_node, fmap_branching):
        fmap_static = self.staticFeature[i]
        fmap_all = {}
        fea_type = {
            's': fmap_static,
            'n': fmap_node,
            'b': fmap_branching
        }
        for fea_type, fmap in fea_type.items():
            for k,v in fmap.items():
                key = "%s_%s" %(fea_type, '0'+str(k) if k < 10 else k)
                fmap_all[key] = v
        return fmap_all

    def get_mean(self, vector):
        if len(vector) == 0:
            return 0
        else:
            return np.mean(vector)

    def get_std(self, vector):
        if len(vector) == 0:
            return 0
        else:
            return np.std(vector)

    def get_quartile(self, vector, percent):
        if len(vector) == 0:
            return 0
        else:
            return np.quantile(vector, percent)


    def add_obj_decrease_ratio_from_parent(self, i, num):
        if i not in self.obj_decrease_ratio_from_parent:
            self.obj_decrease_ratio_from_parent[i] = []
        self.obj_decrease_ratio_from_parent[i].append(num)

    def add_variable_num_branching(self, i):
        if i not in self.num_branching:
            self.num_branching[i] = 0
        self.num_branching[i] += 1


    def get_division(self,a,b):
        if  b == 0:
            return 1000     # use if denominater is zero
        else:
            return a/b
                

    def get_max(self, vector):
        if len(vector) == 0:
            return 0
        else:
            return max(vector)

    def get_min(self, vector):
        if len(vector) == 0:
            return 0
        else:
            return min(vector)
