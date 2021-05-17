from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
from past.utils import old_div
__author__ = 'Ted Ralphs'
__maintainer__ = 'Ted Ralphs (ted@lehigh.edu)'

import pdb
import random, sys, math
try:
    from src.blimpy import PriorityQueue
except ImportError:
    from coinor.blimpy import PriorityQueue
import time
from pulp import LpVariable, lpSum, LpProblem, LpMaximize, LpConstraint
from pulp import LpStatus, value, PULP_CBC_CMD
from .BBTree import BBTree
from .BBTree import MOST_FRACTIONAL, FIXED_BRANCHING, PSEUDOCOST_BRANCHING, STRONG_BRANCHING1, STRONG_BRANCHING2, STRONG_BRANCHING3,\
LEARNED_BRANCHING_GBDT_1, \
LEARNED_BRANCHING_GBDT_2, \
LEARNED_BRANCHING_GBDT_3, \
LEARNED_BRANCHING_ETR_1, \
LEARNED_BRANCHING_ETR_2, \
LEARNED_BRANCHING_ETR_3
from .BBTree import DEPTH_FIRST, BEST_FIRST, BEST_ESTIMATE, INFINITY
import logging
from .strongBranchingFeatures import strongBranchingFeatures
from joblib import load
import numpy as np
from pysmps import smps_loader as smps
import time
import os
def ReadMPS(mps_file):
    # Note MPS file does not specify the sense, we use maximize for all problem
    name, objective_name, constr_names, var_names, var_type, constr_sense, c, A, rhs_names, rhs, bnd_names, bnd = smps.load_mps(mps_file)
    # look https://pypi.org/project/pysmps/ for pysmps api
    if len(rhs_names) != 1:
        print('ERROR: len(rhs_names) != 1')
    RHS = rhs[rhs_names[0]]
    A_new = []
    RHS_new = []
    CONSTRAINTS_new = []
    for i in range(len(constr_sense)):
        if constr_sense[i] == 'E':
            A_new.append(A[i,:])
            A_new.append(-1*A[i,:])
            RHS_new.append(RHS[i])
            RHS_new.append(-1*RHS[i])
            CONSTRAINTS_new.append(constr_names[i]+'_1')
            CONSTRAINTS_new.append(constr_names[i]+'_2')
        elif constr_sense[i] == 'G':
            # use constr <= rhs
            A_new.append(-1*A[i,:])
            RHS_new.append(-1*RHS[i])
            CONSTRAINTS_new.append(constr_names[i])
        elif constr_sense[i] == 'L':
            A_new.append(A[i,:])
            RHS_new.append(RHS[i])
            CONSTRAINTS_new.append(constr_names[i])

    A_new = np.array(A_new)
    RHS_new = np.array(RHS_new)

    OBJ = {}
    MAT = {}
    for j in range(len(var_names)):
        OBJ[var_names[j]] = -c[j]
        MAT[var_names[j]] = A_new[:,j]
    return CONSTRAINTS_new, var_names, OBJ, MAT, RHS_new

def GenerateRandomIP(randomSeed = 1, numVars = 40, maxObjCoeff = 100, maxConsCoeff = 100, numConstrK=20, numConstrEK=1, numConstrP=20, numConstrC=20, numConstrSO = 1):
    # -1 Knapsack ()   # sense >= or <=
    # 0. Equality Knapsack 
    # 1. Packing constraints
    # 2. Coving constraints
    # 3. Special ordered constraints

    random.seed(randomSeed)

    VARIABLES = ["x"+str(i) for i in range(numVars)]

    # OBJ
    OBJ_coef = np.random.randint(1, maxObjCoeff, numVars)
    OBJ_sense = random.random()
    if OBJ_sense < -1:
        OBJ = {VARIABLES[i]:OBJ_coef[i] for i in range(numVars)}    # max
    else:
        OBJ = {VARIABLES[i]: - OBJ_coef[i] for i in range(numVars)} # min

    # Constr
    A = {x: [] for x in VARIABLES}
    RHS = []
    # Knapsack constr
    for j in range(numConstrK):
        coef = np.random.randint(0,maxConsCoeff, numVars)
        rhs = np.random.randint(0,maxConsCoeff)
        sense = random.random()
        if sense < -1:
            for i in range(numVars):
                A[VARIABLES[i]].append(coef[i])    # a^Tx <= b
            RHS.append(rhs)
        else:
            for i in range(numVars):
                A[VARIABLES[i]].append(-coef[i])    # a^Tx >= b
            RHS.append(-rhs)
        
    # Equality Knapsack constr
    for j in range(numConstrEK):
        coef = np.random.randint(0,maxConsCoeff, numVars)
        rhs = np.random.randint(0,maxConsCoeff)
        for i in range(numVars):
            A[VARIABLES[i]].append(coef[i])  
        RHS.append(rhs)
        for i in range(numVars):
            A[VARIABLES[i]].append(-coef[i])  
        RHS.append(-rhs)

    # Packing constr: sum_{j in B} x_j <= 1
    for j in range(numConstrP):
        coef = np.random.randint(0,2, numVars)    
        rhs = 1
        for i in range(numVars):
            A[VARIABLES[i]].append(coef[i])
        RHS.append(rhs)

    # Covering constr: sum_{j in B} x_j >= 1
    for j in range(numConstrC):
        coef = np.random.randint(0,2, numVars)
        rhs = 1
        for i in range(numVars):
            A[VARIABLES[i]].append(-coef[i])
        RHS.append(-rhs)


    # Special Ordered constr sum_{j in B} x_j == 1
    for j in range(numConstrSO):
        coef = np.random.randint(0,2, numVars)
        rhs = 1
        for i in range(numVars):
            A[VARIABLES[i]].append(coef[i])
        RHS.append(rhs)
        for i in range(numVars):
            A[VARIABLES[i]].append(-coef[i])
        RHS.append(-rhs)

    CONSTRAINTS = ['C%s' %(i) for i in range(len(RHS))]
    return CONSTRAINTS, VARIABLES, OBJ, A, RHS

    


def GenerateRandomMIP(numVars = 40, numCons = 20, density = 0.2,
                      maxObjCoeff = 10, maxConsCoeff = 10, 
                      tightness = 2, rand_seed = 2, layout = 'dot', numConstrP =0, numConstrC=0):
    random.seed(rand_seed)
    CONSTRAINTS = ["C"+str(i) for i in range(numCons)]
    if layout == 'dot2tex':
        VARIABLES = ["x_{"+str(i)+"}" for i in range(numVars)]
    else:
        VARIABLES = ["x"+str(i) for i in range(numVars)]
    OBJ = dict((i, random.randint(1, maxObjCoeff)) for i in VARIABLES)
    MAT = dict((i, [random.randint(1, maxConsCoeff)
                    if random.random() <= density else 0
                    for j in CONSTRAINTS]) for i in VARIABLES)
    RHS = [random.randint(int(numVars*density*maxConsCoeff/tightness),
                   int(numVars*density*maxConsCoeff/1.5))
           for i in CONSTRAINTS]

    # Packing constr: sum_{j in B} x_j <= 1
    for j in range(numConstrP):
        coef = (np.random.rand(numVars) < density).astype(int)
        rhs = 1
        for i in range(numVars):
            MAT[VARIABLES[i]].append(coef[i])
        RHS.append(rhs)
        CONSTRAINTS.append('CP%s' %(j))

    # Covering constr: sum_{j in B} x_j >= 1
    for j in range(numConstrC):
        coef = (np.random.rand(numVars) < density).astype(int)
        rhs = 1
        for i in range(numVars):
            MAT[VARIABLES[i]].append(-coef[i])
        RHS.append(-rhs)
        CONSTRAINTS.append('CC%s' %(j))
    return CONSTRAINTS, VARIABLES, OBJ, MAT, RHS


def StrongBranchingLP(T, CONSTRAINTS, VARIABLES, OBJ, MAT, RHS,
        cur_index, parent, relax, branch_var, branch_var_value, sense,rhs, 
        strong_branch_var=None, strong_branch_rhs=None, strong_branch_sense=None, logger=None):
        #====================================
        #    LP Relaxation of strong branching variable i
        #    TODO if slow, change to CYLP
        #====================================
        var   = LpVariable.dicts("", VARIABLES, 0, 1)

        prob = LpProblem("StrongBranchingRelax", LpMaximize)
        prob += lpSum([OBJ[i]*var[i] for i in VARIABLES]), "Objective"
        numCons = len(CONSTRAINTS)
        for j in range(numCons):
            prob += (lpSum([MAT[i][j]*var[i] for i in VARIABLES])<=RHS[j],\
                         CONSTRAINTS[j])
        # Fix all prescribed variables
        if cur_index != 0:
            if sense == '>=':
                prob += LpConstraint(lpSum(var[branch_var]) >= rhs)
            else:
                prob += LpConstraint(lpSum(var[branch_var]) <= rhs)
            pred = parent
            while not str(pred) == '0':
                pred_branch_var = T.get_node_attr(pred, 'branch_var')
                pred_rhs = T.get_node_attr(pred, 'rhs')
                pred_sense = T.get_node_attr(pred, 'sense')
                if pred_sense == '<=':
                    prob += LpConstraint(lpSum(var[pred_branch_var])
                                         <= pred_rhs)
                else:
                    prob += LpConstraint(lpSum(var[pred_branch_var])
                                         >= pred_rhs)
                pred = T.get_node_attr(pred, 'parent')

        # Strong branching 
        if strong_branch_sense == '>=':
            prob += LpConstraint(lpSum(var[strong_branch_var]) >= math.ceil(strong_branch_rhs))
        elif strong_branch_sense == '<=':
            prob += LpConstraint(lpSum(var[strong_branch_var]) <= math.floor(strong_branch_rhs))

        # Solve the LP relaxation
        prob.solve(PULP_CBC_CMD(msg=0))

        # Check infeasibility
        infeasible = LpStatus[prob.status] == "Infeasible" or \
            LpStatus[prob.status] == "Undefined"

        # Print status
        child_relax = 2*relax    # obj of infeasible problem
        if infeasible:
            logger.debug("StrongBranching %s-%s LP Solved, status: Infeasible" %(strong_branch_var, strong_branch_sense))
        else:
            logger.debug("StrongBranching %s-%s LP Solved, status: %s, obj: %s" %(strong_branch_var, strong_branch_sense, 
                LpStatus[prob.status], value(prob.objective)))
            if(LpStatus[prob.status] == "Optimal"):
                child_relax = value(prob.objective)
            else:
                logger.warning("WARNING StrongBranching %s-%s LP STATUS neither infeasible or optimal")
        return child_relax
        

def BranchAndBound(T, CONSTRAINTS, VARIABLES, OBJ, MAT, RHS,
                   branch_strategy = MOST_FRACTIONAL,
                   search_strategy = DEPTH_FIRST,
                   complete_enumeration = False,
                   display_interval = None,
                   binary_vars = True,
                   logger = None,
                   get_strong_branching_feature=False,
                   return_feature = False,
                   timeLimit = 30*60,
                   nodeLimit = 3000,
                   learn_model_prefix = '0_20'
                   ):
    tic = time.time()

    # ----- FOR LEARNED BRANCHING ---------------------
    if branch_strategy.startswith('Learned Branching'):
        name_split = branch_strategy.split(' ')
        learn_model_name = '%s/mdl/%s_fea_strong_%s_%s.joblib' %(os.path.dirname(__file__), name_split[-2].lower(), name_split[-1], learn_model_prefix)
        branching_regr = load(learn_model_name)
        get_strong_branching_feature=True
    # ----------------- END ---------------------------


    # ----- FOR STORE STRONG BRANCHING FEATURE ---------
    if return_feature:
        features = []

    if get_strong_branching_feature:
        feaGenerator = strongBranchingFeatures(CONSTRAINTS, VARIABLES, OBJ, MAT, RHS)
    # ------------------ END ---------------------------

    if T.get_layout() == 'dot2tex':
        cluster_attrs = {'name':'Key', 'label':r'\text{Key}', 'fontsize':'12'}
        T.add_node('C', label = r'\text{Candidate}', style = 'filled',
                      color = 'yellow', fillcolor = 'yellow')
        T.add_node('I', label = r'\text{Infeasible}', style = 'filled',
                      color = 'orange', fillcolor = 'orange')
        T.add_node('S', label = r'\text{Solution}', style = 'filled',
                      color = 'lightblue', fillcolor = 'lightblue')
        T.add_node('P', label = r'\text{Pruned}', style = 'filled',
                      color = 'red', fillcolor = 'red')
        T.add_node('PC', label = r'\text{Pruned}$\\ $\text{Candidate}', style = 'filled',
                      color = 'red', fillcolor = 'yellow')
    else:
        cluster_attrs = {'name':'Key', 'label':'Key', 'fontsize':'12'}
        T.add_node('C', label = 'Candidate', style = 'filled',
                      color = 'yellow', fillcolor = 'yellow')
        T.add_node('I', label = 'Infeasible', style = 'filled',
                      color = 'orange', fillcolor = 'orange')
        T.add_node('S', label = 'Solution', style = 'filled',
                      color = 'lightblue', fillcolor = 'lightblue')
        T.add_node('P', label = 'Pruned', style = 'filled',
                      color = 'red', fillcolor = 'red')
        T.add_node('PC', label = 'Pruned \n Candidate', style = 'filled',
                      color = 'red', fillcolor = 'yellow')
    T.add_edge('C', 'I', style = 'invisible', arrowhead = 'none')
    T.add_edge('I', 'S', style = 'invisible', arrowhead = 'none')
    T.add_edge('S', 'P', style = 'invisible', arrowhead = 'none')
    T.add_edge('P', 'PC', style = 'invisible', arrowhead = 'none')
    T.create_cluster(['C', 'I', 'S', 'P', 'PC'], cluster_attrs)
    # The initial lower bound
    LB = -INFINITY
    # The number of LP's solved, and the number of nodes solved
    node_count = 1
    iter_count = 0
    lp_count = 0
    
    if binary_vars:
        var   = LpVariable.dicts("", VARIABLES, 0, 1)
    else:
        var   = LpVariable.dicts("", VARIABLES)
    
    numCons = len(CONSTRAINTS)
    numVars = len(VARIABLES)
    # List of incumbent solution variable values
    opt = dict([(i, 0) for i in VARIABLES])
    pseudo_u = dict((i, (OBJ[i], 0)) for i in VARIABLES)
    pseudo_d = dict((i, (OBJ[i], 0)) for i in VARIABLES)
    logger.info("===========================================")
    logger.info("Starting Branch and Bound")
    if branch_strategy == MOST_FRACTIONAL:
        logger.info("Most fractional variable")
    elif branch_strategy == FIXED_BRANCHING:
        logger.info("Fixed order")
    elif branch_strategy == PSEUDOCOST_BRANCHING:
        logger.info("Pseudocost brancing")
    else:
        logger.info("Unknown branching strategy %s" %branch_strategy)
    if search_strategy == DEPTH_FIRST:
        logger.info("Depth first search strategy")
    elif search_strategy == BEST_FIRST:
        logger.info("Best first search strategy")
    else:
        logger.info("Unknown search strategy %s" %search_strategy)
    logger.info("===========================================")
    # List of candidate nodes
    Q = PriorityQueue()
    # The current tree depth
    cur_depth = 0
    cur_index = 0
    # Timer
    timer = time.time()
    Q.push(0, -INFINITY, (0, None, None, None, None, None, None))
    # Branch and Bound Loop
    while not Q.isEmpty():
        now = time.time()
        if now - tic > timeLimit:
            break
        if node_count > nodeLimit:
            break
        infeasible = False
        integer_solution = False
        (cur_index, parent, relax, branch_var, branch_var_value, sense,
        rhs) = Q.pop()
        if cur_index != 0:
            cur_depth = T.get_node_attr(parent, 'level') + 1
        else:
            cur_depth = 0
        logger.info("")
        logger.info("----------------------------------------------------")
        logger.info("")
        if LB > -INFINITY:
            logger.info("Node: %s, Depth: %s, LB: %s" %(cur_index,cur_depth,LB))
        else:
            logger.info("Node: %s, Depth: %s, LB: %s" %(cur_index,cur_depth,"None"))
        if relax is not None and relax <= LB:
            logger.info("Node pruned immediately by bound")
            T.set_node_attr(parent, 'color', 'red')
            continue
        #====================================
        #    LP Relaxation
        #====================================
        # Compute lower bound by LP relaxation
        prob = LpProblem("relax", LpMaximize)
        prob += lpSum([OBJ[i]*var[i] for i in VARIABLES]), "Objective"
        for j in range(numCons):
            prob += (lpSum([MAT[i][j]*var[i] for i in VARIABLES])<=RHS[j],\
                         CONSTRAINTS[j])
        # Fix all prescribed variables
        branch_vars = []
        if cur_index != 0:
            #sys.stdout.write("Branching variables: ")
            logger.info("Branching variables: ")
            branch_vars.append(branch_var)
            if sense == '>=':
                prob += LpConstraint(lpSum(var[branch_var]) >= rhs)
            else:
                prob += LpConstraint(lpSum(var[branch_var]) <= rhs)
            logger.info(branch_var)
            pred = parent
            while not str(pred) == '0':
                pred_branch_var = T.get_node_attr(pred, 'branch_var')
                pred_rhs = T.get_node_attr(pred, 'rhs')
                pred_sense = T.get_node_attr(pred, 'sense')
                if pred_sense == '<=':
                    prob += LpConstraint(lpSum(var[pred_branch_var])
                                         <= pred_rhs)
                else:
                    prob += LpConstraint(lpSum(var[pred_branch_var])
                                         >= pred_rhs)
                logger.info(pred_branch_var)
                branch_vars.append(pred_branch_var)
                pred = T.get_node_attr(pred, 'parent')
            logger.info('')

        # Solve the LP relaxation
        prob.solve(PULP_CBC_CMD(msg=0))
        lp_count = lp_count +1
        # Check infeasibility
        infeasible = LpStatus[prob.status] == "Infeasible" or \
            LpStatus[prob.status] == "Undefined"
        # Print status
        if infeasible:
            logger.info("LP Solved, status: Infeasible")
        else:
            logger.info("LP Solved, status: %s, obj: %s" %(LpStatus[prob.status],
                                                     value(prob.objective)))
        if(LpStatus[prob.status] == "Optimal"):
            relax = value(prob.objective)
            # Update pseudocost
            if branch_var != None:
                if sense == '<=':
                    pseudo_d[branch_var] = (
                    old_div((pseudo_d[branch_var][0]*pseudo_d[branch_var][1] +
                    old_div((T.get_node_attr(parent, 'obj') - relax),
                    (branch_var_value - rhs))),(pseudo_d[branch_var][1]+1)),
                    pseudo_d[branch_var][1]+1)
                else:
                    pseudo_u[branch_var] = (
                    old_div((pseudo_u[branch_var][0]*pseudo_d[branch_var][1] +
                     old_div((T.get_node_attr(parent, 'obj') - relax),
                     (rhs - branch_var_value))),(pseudo_u[branch_var][1]+1)),
                    pseudo_u[branch_var][1]+1)
            var_values = dict([(i, var[i].varValue) for i in VARIABLES])


            #  ---- FOR STRONG BRANCHING NODE FEATURES --------
            if get_strong_branching_feature: 

                # reduced cost of variables
                var_reduced_cost = dict([(i, var[i].dj) for i in VARIABLES])

                frac_to_ceil =  dict([(i, math.ceil(var_values[i]) - var_values[i]) for i in VARIABLES])
                frac_to_floor =  dict([(i, var_values[i] - math.floor(var_values[i])) for i in VARIABLES])
                n_fixed_vars = len(branch_vars)
                fmap_node = feaGenerator.getNodeFeature(var_reduced_cost, frac_to_ceil, frac_to_floor, n_fixed_vars) 
                
                # relate to obj
                if parent and T.get_node_attr(parent, 'obj'):
                    ratio = (T.get_node_attr(parent, 'obj') - relax)/relax
                    feaGenerator.add_obj_decrease_ratio_from_parent(branch_var, ratio)
            # ---- END FOR STRONG BRANCHING NODE FEATURES ------

            integer_solution = 1
            for i in VARIABLES:
                if (abs(round(var_values[i]) - var_values[i]) > .001):
                    integer_solution = 0
                    break
            # Determine integer_infeasibility_count and
            # Integer_infeasibility_sum for scatterplot and such
            integer_infeasibility_count = 0
            integer_infeasibility_sum = 0.0
            for i in VARIABLES:
                if (var_values[i] not in set([0,1])):
                    integer_infeasibility_count += 1
                    integer_infeasibility_sum += min([var_values[i],
                                                      1.0-var_values[i]])
            if (integer_solution and relax>LB):
                LB = relax
                for i in VARIABLES:
                    # These two have different data structures first one
                    #list, second one dictionary
                    opt[i] = var_values[i]
                logger.info("New best solution found, objective: %s" %relax)
                for i in VARIABLES:
                    if var_values[i] > 0:
                        logger.info("%s = %s" %(i, var_values[i]))
            elif (integer_solution and relax<=LB):
                logger.info("New integer solution found, objective: %s" %relax)
                for i in VARIABLES:
                    if var_values[i] > 0:
                        logger.info("%s = %s" %(i, var_values[i]))
            else:
                logger.info("Fractional solution:")
                for i in VARIABLES:
                    if var_values[i] > 0:
                        logger.info("%s = %s" %(i, var_values[i]))
            #For complete enumeration
            if complete_enumeration:
                relax = LB - 1
        else:
            relax = INFINITY
        if integer_solution:
            logger.info("Integer solution")
            BBstatus = 'S'
            status = 'integer'
            color = 'lightblue'
        elif infeasible:
            logger.info("Infeasible node")
            BBstatus = 'I'
            status = 'infeasible'
            color = 'orange'
        elif not complete_enumeration and relax <= LB:
            logger.info("Node pruned by bound (obj: %s, UB: %s)" %(relax,LB))
            BBstatus = 'P'
            status = 'fathomed'
            color = 'red'
        elif cur_depth >= numVars :
            logger.info("Reached a leaf")
            BBstatus = 'fathomed'
            status = 'L'
        else:
            BBstatus = 'C'
            status = 'candidate'
            color = 'yellow'
        if BBstatus == 'I':
            if T.get_layout() == 'dot2tex':
                label = r'\text{I}'
            else:
                label = 'I'
        else:
            label = "%.1f"%relax
        if iter_count == 0:
            if status != 'candidate':
                integer_infeasibility_count = None
                integer_infeasibility_sum = None
            if status == 'fathomed':
                if T._incumbent_value is None:
                    logger.warning('WARNING: Encountered "fathom" line before '+\
                        'first incumbent.')
            T.AddOrUpdateNode(0, None, None, 'candidate', relax,
                             integer_infeasibility_count,
                             integer_infeasibility_sum,
                             label = label,
                             obj = relax, color = color,
                             style = 'filled', fillcolor = color)
            if status == 'integer':
                T._previous_incumbent_value = T._incumbent_value
                T._incumbent_value = relax
                T._incumbent_parent = -1
                T._new_integer_solution = True
#           #Currently broken
#           if ETREE_INSTALLED and T.attr['display'] == 'svg':
#               T.write_as_svg(filename = "node%d" % iter_count,
#                                 nextfile = "node%d" % (iter_count + 1),
#                                 highlight = cur_index)
        else:
            _direction = {'<=':'L', '>=':'R'}
            if status == 'infeasible':
                integer_infeasibility_count = T.get_node_attr(parent,
                                     'integer_infeasibility_count')
                integer_infeasibility_sum = T.get_node_attr(parent,
                                     'integer_infeasibility_sum')
                relax = T.get_node_attr(parent, 'lp_bound')
            elif status == 'fathomed':
                if T._incumbent_value is None:
                    logger.warning('WARNING: Encountered "fathom" line before'+\
                        ' first incumbent.')
                    logger.info('  This may indicate an error in the input file.')
            elif status == 'integer':
                integer_infeasibility_count = None
                integer_infeasibility_sum = None
            T.AddOrUpdateNode(cur_index, parent, _direction[sense],
                                 status, relax,
                                 integer_infeasibility_count,
                                 integer_infeasibility_sum,
                                 branch_var = branch_var,
                                 branch_var_value = var_values[branch_var],
                                 sense = sense, rhs = rhs, obj = relax,
                                 color = color, style = 'filled',
                                 label = label, fillcolor = color)
            if status == 'integer':
                T._previous_incumbent_value = T._incumbent_value
                T._incumbent_value = relax
                T._incumbent_parent = parent
                T._new_integer_solution = True
            # Currently Broken
#           if ETREE_INSTALLED and T.attr['display'] == 'svg':
#               T.write_as_svg(filename = "node%d" % iter_count,
#                                 prevfile = "node%d" % (iter_count - 1),
#                                 nextfile = "node%d" % (iter_count + 1),
#                                 highlight = cur_index)
            if T.get_layout() == 'dot2tex':
                _dot2tex_label = {'>=':' \geq ', '<=':' \leq '}
                T.set_edge_attr(parent, cur_index, 'label',
                                   str(branch_var) + _dot2tex_label[sense] +
                                   str(rhs))
            else:
                T.set_edge_attr(parent, cur_index, 'label',
                                   str(branch_var) + sense + str(rhs))
        iter_count += 1
        if BBstatus == 'C':
            # Branching:
            # Choose a variable for branching
            branching_var = None
            if branch_strategy == FIXED_BRANCHING:
                #fixed order
                for i in VARIABLES:
                    frac = min(var[i].varValue-math.floor(var[i].varValue),
                               math.ceil(var[i].varValue) - var[i].varValue)
                    if (frac > 0):
                        min_frac = frac
                        branching_var = i
                        # TODO(aykut): understand this break
                        break
            elif branch_strategy == MOST_FRACTIONAL:
                #most fractional variable
                min_frac = -1
                for i in VARIABLES:
                    frac = min(var[i].varValue-math.floor(var[i].varValue),
                               math.ceil(var[i].varValue)- var[i].varValue)
                    if (frac> min_frac):
                        min_frac = frac
                        branching_var = i
            elif branch_strategy == PSEUDOCOST_BRANCHING:
                scores = {}
                for i in VARIABLES:
                    # find the fractional solutions
                    if (var[i].varValue - math.floor(var[i].varValue)) != 0:
                        scores[i] = min(pseudo_u[i][0]*(1-var[i].varValue),
                                        pseudo_d[i][0]*var[i].varValue)
                    # sort the dictionary by value
                branching_var = sorted(list(scores.items()),
                                       key=lambda x : x[1])[-1][0]
            elif branch_strategy.startswith('Learned Branching'):
                scores = {}
                for i in VARIABLES:
                    # find the fractional solutions
                    if (var_values[i] - math.floor(var_values[i])) != 0:
                        fmap_branching = feaGenerator.getBranchingFeature(i)
                        fmap_all = feaGenerator.getFeature(i, fmap_node[i], fmap_branching)
                        X_fea = [fmap_all[f_name] for f_name in feaGenerator.fea_col] 
                        score = branching_regr.predict(np.reshape(X_fea, (1, len(feaGenerator.fea_col))))
                        scores[i] = score
                branching_var = sorted(list(scores.items()),
                                       key=lambda x : x[1])[-1][0]

            elif branch_strategy.startswith('Strong Branching'):
                scores = {}
                for i in VARIABLES:
                    # find the fractional solutions
                    if (var_values[i] - math.floor(var_values[i])) != 0:
                        # strong branching LP obj of left child
                        left_relax = StrongBranchingLP(T, CONSTRAINTS, VARIABLES, OBJ, MAT, RHS,
                                cur_index, parent, relax, branch_var, branch_var_value, sense,rhs,
                                strong_branch_var=i, strong_branch_rhs=var_values[i], strong_branch_sense="<=", logger=logger)
                        left_diff_abs = abs((left_relax - relax)/relax)
                        # strong branching LP obj of right child
                        right_relax = StrongBranchingLP(T, CONSTRAINTS, VARIABLES, OBJ, MAT, RHS, 
                                cur_index, parent, relax, branch_var, branch_var_value, sense,rhs,
                                strong_branch_var=i, strong_branch_rhs=var_values[i], strong_branch_sense=">=", logger=logger)
                        right_diff_abs = abs((right_relax - relax)/relax)
                        lp_count += 2
                        if branch_strategy == STRONG_BRANCHING1:
                            score = left_diff_abs*right_diff_abs  # score from paper Alejandro, et al., 2017 # TODO normalize it
                        elif branch_strategy == STRONG_BRANCHING2:
                            score = min(left_diff_abs,right_diff_abs)
                        elif branch_strategy == STRONG_BRANCHING3:
                            score = left_diff_abs + right_diff_abs

                        scores[i] = score
                # NOTE: this will choose, 
                # 1. if exists one varible, both children are infeasible, three strategies choose this variable
                # 2. esif exists one variable, one of children is infeasible,three strategies choose this variable
                # 3. else score could be different for three score strategies.

                branching_var = sorted(list(scores.items()),
                                       key=lambda x : x[1])[-1][0]

                # -------- FOR STRONG BRANCHING BRANCHING FEATURES ---------
                if get_strong_branching_feature:
                    for i, score in scores.items():
                        fmap_branching = feaGenerator.getBranchingFeature(i)                      
                        fmap_all = feaGenerator.getFeature(i, fmap_node[i], fmap_branching)
                        fmap_all['score'] = score
                        if return_feature:
                            features.append(fmap_all)
                # ----------------------- END ------------------------------


            else:
                logger.info("Unknown branching strategy %s" %branch_strategy)
                exit()
            if branching_var is not None:
                logger.info("Branching on variable %s" %branching_var)
                if get_strong_branching_feature:
                    feaGenerator.add_variable_num_branching(branching_var)
            #Create new nodes
            if search_strategy == DEPTH_FIRST:
                priority = (-cur_depth - 1, -cur_depth - 1)
            elif search_strategy == BEST_FIRST:
                priority = (-relax, -relax)
            elif search_strategy == BEST_ESTIMATE:
                priority = (-relax - pseudo_d[branching_var][0]*\
                                 (math.floor(var[branching_var].varValue) -\
                                      var[branching_var].varValue),
                            -relax + pseudo_u[branching_var][0]*\
                                 (math.ceil(var[branching_var].varValue) -\
                                      var[branching_var].varValue))
            node_count += 1
            Q.push(node_count, priority[0], (node_count, cur_index, relax, branching_var,
                    var_values[branching_var],
                    '<=', math.floor(var[branching_var].varValue)))
            node_count += 1
            Q.push(node_count, priority[1], (node_count, cur_index, relax, branching_var,
                    var_values[branching_var],
                    '>=', math.ceil(var[branching_var].varValue)))
            T.set_node_attr(cur_index, color, 'green')
        if T.root is not None and display_interval is not None and\
                iter_count%display_interval == 0:
            T.display(count=iter_count)

    timer = int(math.ceil((time.time()-timer)*1000))
    logger.info("")
    logger.info("===========================================")
    logger.info("Branch and bound completed in %sms" %timer)
    logger.info("Strategy: %s" %branch_strategy)
    if complete_enumeration:
        logger.info("Complete enumeration")
    logger.info("%s nodes visited " %node_count)
    logger.info("%s LP's solved" %lp_count)
    logger.info("===========================================")
    if Q.isEmpty():
        logger.info("Optimal solution")
        status = 'optimal'
    else:
        logger.info("Terminate by time or node limit")
        UB = []
        while not Q.isEmpty():
            (cur_index, parent, relax, branch_var, branch_var_value, sense,
            rhs) = Q.pop() 
            UB.append(relax)
        UB = max(UB)
        status = 'gap-%.2f' %((UB - LB)/ LB)
    #logger.info optimal solution
    for i in sorted(VARIABLES):
        if opt[i] > 0:
            logger.info("%s = %s" %(i, opt[i]))
    logger.info("Objective function value")
    logger.info(LB)
    logger.info("===========================================")
    if T.attr['display'] != 'off':
        T.display(count=iter_count)
    T._lp_count = lp_count
    res = {
        'nNode': node_count,
        'Time': timer/1000,
        'lpCount': lp_count,
        'status': status
    }
    if return_feature:
        res['strongBranchingFea'] = features
    return opt, LB, res

if __name__ == '__main__':    
    T = BBTree()
    T.set_layout('dot')
    #T.set_layout('dot2tex')
    #T.set_display_mode('file')
    #T.set_display_mode('xdot')
    T.set_display_mode('matplotlib')
    CONSTRAINTS, VARIABLES, OBJ, MAT, RHS = GenerateRandomMIP(numVars = 15,
                                                              numCons = 5,
                                                              rand_seed = 120)
    BranchAndBound(T, CONSTRAINTS, VARIABLES, OBJ, MAT, RHS,
                   branch_strategy = MOST_FRACTIONAL,
                   search_strategy = BEST_FIRST,
                   display_interval = 1)

