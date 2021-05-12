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
                0:'sign_of_ci',
                1:'ratio_ci_sum_pos_ci',
                2:'ratio_ci_sum_neg_ci',
                3:'max_M_j1_plus',
                4:'min_M_j1_plus',
                5:'max_M_j1_minus',
                6:'min_M_j1_minus',
                7:'max_M_j2_plus',
                8:'min_M_j2_plus',
                9:'max_M_j2_minus',
                10:'min_M_j2_minus',
                11:'max_M_j3_plus',
                12:'min_M_j3_plus',
                13:'max_M_j3_minus',
                14:'min_M_j3_minus',

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

            # sign of ci
            if ci < 0:
                fmap[0] = -1
            elif ci == 0:
                fmap[0] = 0
            elif ci > 0:
                fmap[0] = 1

            # |ci| / sum of |ci| such that ci >= 0
            sum_pos_ci = sum([abs(v) for v in c.values() if v >= 0])
            sum_neg_ci = sum([abs(v) for v in c.values() if v < 0])
            fmap[1] = self.get_division(abs(ci),sum_pos_ci)
            fmap[2] = self.get_division(abs(ci),sum_neg_ci)
            
            # M1
            M_j1_plus = [self.get_division(Ai[j],abs(b[j])) for j in range(nConstr) if b[j] >= 0]
            M_j1_minus = [self.get_division(Ai[j],abs(b[j])) for j in range(nConstr) if b[j] < 0]
            fmap[3] = self.get_max(M_j1_plus)
            fmap[4] = self.get_min(M_j1_plus)
            fmap[5] = self.get_max(M_j1_minus)
            fmap[6] = self.get_min(M_j1_minus)
        
            # M2
            M_j2_plus =  [self.get_division(abs(ci),Ai[j]) for j in range(nConstr) if ci >= 0]
            M_j2_minus = [self.get_division(abs(ci),Ai[j]) for j in range(nConstr) if ci < 0]
            fmap[7] = self.get_max(M_j2_plus)
            fmap[8] = self.get_min(M_j2_plus)
            fmap[9] = self.get_max(M_j2_minus)
            fmap[10] = self.get_min(M_j2_minus)

            # M3
            sum_pos_Ajk = []
            sum_neg_Ajk = []
            for j in range(nConstr):
                sum_pos_Ajk.append(sum([abs(self.MAT[k][j]) for k in self.VARIABLES if self.MAT[k][j] >= 0]))
                sum_neg_Ajk.append(sum([abs(self.MAT[k][j]) for k in self.VARIABLES if self.MAT[k][j] < 0]))
            M_j3_plus =  [self.get_division(abs(Ai[j]),sum_pos_Ajk[j]) for j in range(nConstr)]
            M_j3_minus = [self.get_division(abs(Ai[j]),sum_neg_Ajk[j]) for j in range(nConstr)]
            # NOTE, miss M_j3_plusplus and M_j3_plusminus, etc., 
            fmap[11] = self.get_max(M_j3_plus)
            fmap[12] = self.get_min(M_j3_plus)
            fmap[13] = self.get_max(M_j3_minus)
            fmap[14] = self.get_min(M_j3_minus)

            fmap_static[i] = fmap
        return fmap_static

    def getNodeFeature(self, var_reduced_cost, frac_to_ceil, frac_to_floor, n_fixed_vars):
        fmap_node = {}
        for i in self.VARIABLES:
            fmap = {}
            fmap[0] = var_reduced_cost[i]/abs(self.OBJ[i])
            fmap[1] = frac_to_ceil[i]
            fmap[2] = frac_to_floor[i]
            fmap[3] = n_fixed_vars/len(self.VARIABLES)
            fmap_node[i] = fmap
        return fmap_node

    def getBranchingFeature(self, i):
        fmap = {}
        fmap[0] = self.get_max(self, self.obj_decrease_ratio_from_parent[i])
        fmap[1] = self.get_min(self, self.obj_decrease_ratio_from_parent[i])
        fmap[2] = self.get_mean(self, self.obj_decrease_ratio_from_parent[i])
        fmap[3] = self.get_std(self, self.obj_decrease_ratio_from_parent[i])
        fmap[4] = self.get_quartile(self, self.obj_decrease_ratio_from_parent[i], 0.25)
        fmap[5] = self.get_quartile(self, self.obj_decrease_ratio_from_parent[i], 0.5)
        fmap[6] = self.get_quartile(self, self.obj_decrease_ratio_from_parent[i], 0.75)
        fmap[7] = self.num_branching[i]/sum(self.num_branching.values())
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
