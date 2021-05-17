import random, sys, math
try:
    from src.blimpy import PriorityQueue
except ImportError:
    from coinor.blimpy import PriorityQueue
import time
from pulp import LpVariable, lpSum, LpProblem, LpMaximize, LpConstraint
from pulp import LpStatus, value
from coinor.grumpy.BBTree import BBTree
from coinor.grumpy.BBTree import MOST_FRACTIONAL, FIXED_BRANCHING, PSEUDOCOST_BRANCHING, STRONG_BRANCHING1, STRONG_BRANCHING2, STRONG_BRANCHING3,\
LEARNED_BRANCHING_GBDT_1, \
LEARNED_BRANCHING_GBDT_2, \
LEARNED_BRANCHING_GBDT_3, \
LEARNED_BRANCHING_ETR_1, \
LEARNED_BRANCHING_ETR_2, \
LEARNED_BRANCHING_ETR_3
from coinor.grumpy.BBTree import DEPTH_FIRST, BEST_FIRST, BEST_ESTIMATE, INFINITY
from coinor.grumpy.BranchAndBound import GenerateRandomMIP, BranchAndBound, ReadMPS
import os
from tqdm import tqdm
import logging
import pandas as pd

def get_logger(log_file, log_level):

    logger = logging.getLogger()
    if log_level == 'warning':
        logger.setLevel(level = logging.WARNING)
    elif log_level == 'info':
        logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

if __name__ == '__main__':

    # ---------- revise below variables, to set up the test --------

    branching_type = 'all'                  # just to name the output, can be any thing
    case_type = 'random'                    # 'mps' or random, mps: mps instances, random: randomly generated instances
    N_start = 0                             # random seed from, work if case_type == 'random'
    N_end   = 20                            # random seed end, work if case_type == 'random'
    return_feature = False                  # if true, return the strong branching features
    branches = [                            # branching methods, will enumerate
        STRONG_BRANCHING1,
        LEARNED_BRANCHING_GBDT_2]   
    learn_model_prefix = '0_20'             # '0_20': model trained from sample from knapsack constrained problems;
                                            # 'randomKPC_0_10': model trained from knapsack constrained, packing, covering constained problems

    # ----------- END, just run this file ---------------------------


    log_file = '%s/%s' %(os.getcwd(), 'log_%s.txt' %(branching_type))
    log_level = 'warning'
    logger = get_logger(log_file, log_level)

    res_all = []
    fea_strong_1 = []
    fea_strong_2 = []
    fea_strong_3 = []

    mpsfiles = [
        'p0033',
        'p0201',
        'stein27',
        'lseu',
        'mod008',
    ]

    if case_type.startswith('mps'):
        cases = mpsfiles
    else:
        cases = range(N_start, N_end)

    for i in tqdm(cases):
        if case_type.startswith('mps'):
            CONSTRAINTS, VARIABLES, OBJ, MAT, RHS = ReadMPS('miplib3/%s' %(i))
        else:
            random.seed(i)
            numVars = random.randint(40,100)
            numCons = random.randint(20,60)
            maxObjCoeff = random.randint(1, 100)
            maxConsCoeff = random.randint(1, 100)
            
            numConstrP = 0 #random.randint(20,40)
            numConstrC = 0 #random.randint(0,20)
            
            CONSTRAINTS, VARIABLES, OBJ, MAT, RHS = GenerateRandomMIP(numVars = numVars,
                                                                      numCons = numCons,
                                                                      maxObjCoeff = maxObjCoeff,
                                                                      maxConsCoeff = maxConsCoeff,
                                                                      rand_seed = i,
                                                                      density = 0.2,
                                                                      numConstrP = numConstrP,
                                                                      numConstrC = numConstrC
                                                                      )
        for branch in branches:
            T = BBTree()
            T.set_layout('dot')
            T.set_display_mode('off')
            if branch in [STRONG_BRANCHING1, STRONG_BRANCHING2, STRONG_BRANCHING3,]:
                opt, LB, res = BranchAndBound(T, CONSTRAINTS, VARIABLES, OBJ, MAT, RHS,
                            branch_strategy = branch,
                            search_strategy = BEST_FIRST,
                            display_interval = 100,
                            complete_enumeration = False,
                            logger = logger,
                            nodeLimit=20000,
                            timeLimit = 15*60,
                            get_strong_branching_feature = return_feature,
                            return_feature = return_feature
                            )
            else:
                opt, LB, res = BranchAndBound(T, CONSTRAINTS, VARIABLES, OBJ, MAT, RHS,
                            branch_strategy = branch,
                            search_strategy = BEST_FIRST,
                            display_interval = 100,
                            complete_enumeration = False,
                            logger = logger,
                            nodeLimit=20000,
                            timeLimit = 15*60,
                            learn_model_prefix = learn_model_prefix)
            if return_feature:
                if branch == STRONG_BRANCHING1:
                    fea_strong_1 += res['strongBranchingFea']
                    res.pop('strongBranchingFea')
                if branch == STRONG_BRANCHING2:
                    fea_strong_2 += res['strongBranchingFea']
                    res.pop('strongBranchingFea')
                if branch == STRONG_BRANCHING3:
                    fea_strong_3 += res['strongBranchingFea']
                    res.pop('strongBranchingFea')
            res['branching'] = branch       
            if not case_type.startswith('mps'):
                res['case'] = '%s_v%s_c%s' %(i, numVars, numCons)
            else:
                res['case'] = i
            res['obj'] = LB
            print('case=%4s, obj=%6s, branching=%20s, node=%5s, time=%10s, lpCount=%5s, status=%10s' %(res['case'], res['obj'], res['branching'], res['nNode'], res['Time'], res['lpCount'], res['status']))
            res_all.append(res)
    df_res = pd.DataFrame(res_all)
    if not case_type.startswith('mps'):
        case_type = case_type + '_' + str(N_start)+'_'+str(N_end)
    df_res.to_csv('metrics_branching_method_%s_%s_cases.csv' %(case_type, branching_type), index=None)

    if return_feature:
        df_fea1 = pd.DataFrame(fea_strong_1)
        df_fea1.to_csv('fea_strong_1_%s.csv' %(case_type), index=None)
        df_fea2 = pd.DataFrame(fea_strong_2)
        df_fea2.to_csv('fea_strong_2_%s.csv' %(case_type), index=None)
        df_fea3 = pd.DataFrame(fea_strong_3)
        df_fea3.to_csv('fea_strong_3_%s.csv' %(case_type), index=None)
