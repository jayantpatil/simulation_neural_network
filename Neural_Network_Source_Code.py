import random
import math

tolerance_error = 0.05
alpha = 0.3
threshold = 0.0
cluster_count = 0
parameter_assigned = dict()
member = list()
network_dimensions = list()
start_nodes= list()
end_nodes =list()
prev_vector = list()
vector = list()
min_dist_index = list()
v_center  = list()
distance = list()
nodes = list()
training = list()
checking = list()
end_member_check = dict()
network = dict()
path_weights = dict()
error_at_nodes = dict()
node_values = dict()
data_parmeter_count = 0

def neuron_function(x):
    return 1/(1+math.exp(-x))

def neuron_function_dash(x):
    return neuron_function(x)*(1-neuron_function(x))

def reset_node_values():
    for i in nodes:
        node_values[i] = 1.0

def reset_error_values():
    for i in nodes:
        if i in start_nodes:
            error_at_nodes[i] = 0.0
        else:
            error_at_nodes[i] = 1.0
          
def assign_input(node,value):
    node_values[node] = value

def calculate_node_value(x):
    temp = 0.0
    if x not in start_nodes:
        temp = 0.0
        for i in (network[x])[0]:
            temp = temp + (node_values[i]*path_weights[str(i)+"_"+str(x)])
        node_values[x] =  round(neuron_function(temp - threshold),6)
        
def calculate_output():
    for i in (sorted(network.keys())):
        calculate_node_value(i)

def pass_values_to_network(i):
    for j in (start_nodes):
        assign_input(j, training[parameter_assigned[j]][i])
    calculate_output()
        
def pass_check_values_to_network(i): 
    for j in (start_nodes):
        assign_input(j, checking[parameter_assigned[j]][i])
    calculate_output()
   
def total_error (i):
    for j in end_nodes:
        error_at_nodes[j] = member[i][end_member_check[j]]  - node_values[j]
        
def error_at_node():
    for i in reversed(sorted(error_at_nodes.keys())):
        if i not in end_nodes and i not in start_nodes:
            temp = 0.0
            for j in network[i][1]:
                temp = temp+ path_weights[str(i)+"_"+str(j)] * error_at_nodes[j]
            error_at_nodes[i] = round(node_values[i]*(1-node_values[i])*temp,6)

def update_path_weight(x,y):
    if str(x)+'_'+str(y) in path_weights:
        path_weights[str(x)+'_'+str(y)] = round(path_weights[str(x)+'_'+str(y)] + (alpha* error_at_nodes[y]*node_values[x]),6)
    else :
        print "Invalid path"
        return False

def update_all_path_weights():
    for i in network.keys():
        for j in network[i][1]:
            update_path_weight(i,j)

def accept():
    global  tolerance_error ,alpha,threshold,total_layers,network_dimensions, start_nodes, end_nodes,cluster_count,data_parmeter_count , training
    tolerance_error = float(raw_input())
    alpha = float(raw_input())
    threshold = float(raw_input())
    total_layers = int(raw_input())
    network_dimensions = [int (i) for i in raw_input().split()]
    ini_node = 11
    curr_node = ini_node
    for i in range(total_layers):
        for j in range(network_dimensions[i]):
            nodes.append(curr_node)  
            curr_node = curr_node + 1
        curr_node = ini_node + ((i+1)*10)
    create_network()
    start_nodes = [i+ini_node for i in range((network_dimensions[0]))]   
    net_dim_len = len(network_dimensions)
    end_nodes = [i+(net_dim_len*10+1) for i in range(network_dimensions[net_dim_len-1])]
    assign_path_weights()
    reset_node_values()
    reset_error_values()
    cluster_count = len(end_nodes)
    data_parmeter_count = int (raw_input())
    for i in range(data_parmeter_count):
        training.append( [ float (i) for i in raw_input().split() ])
    for i in range(data_parmeter_count):
        checking.append( [ float (i) for i in raw_input().split() ])
  
def assign_path_weights():
    k = 0
    x =0
    weights = [float (i) for i in raw_input().split()]
    for i in range(len(network_dimensions)-1): 
        x = x+(network_dimensions[i]*network_dimensions[i+1])
    if len(weights) < x:
        raise ValueError("Provide Sufficient path weights ")

    for i in sorted(network.keys()):
        for j in network[i][1]:
            path_weights[str(i)+"_"+str(j)] = weights[k]
            k = k+1
   
def create_network():
    for i in nodes:
        network[i] = [list(),list()]
    for i in nodes:
        for j in nodes:
            if (i/10) == ((j/10)+1):
                network[i][0].append(j)
            elif (i/10) == ((j/10)-1):
                network[i][1].append(j)

def is_err_not_minimum():
    flag = False
    for i in end_nodes:
        if abs(error_at_nodes[i]) > tolerance_error:
            flag = True
    return flag 

def create_cluster (c, lists):
    global vector,v_center,member
    create_initialvector(c,lists)
    while not is_prev_vector_same():
        get_cluster_centers(c,lists)
        get_dist_from_clu_center(c,lists)
        calculate_min_distance_index()
        update_vector(c,lists)
    member = [list(i) for i in zip(*vector )]

def is_prev_vector_same():
    return prev_vector==vector

def update_vector(c,lists):
    global vector, prev_vector,min_dist_index, v_center , distance 
    v_center = list()
    prev_vector = vector
    vector  = list()
    n = len(lists[0])
    for i in range (c):
        vector.append([])
        for j in range(n):
            vector[i].append(0)
    for i in range(n):
        vector[min_dist_index[i]][i] = 1
    min_dist_index = list()   

def calculate_min_distance_index():
    global min_dist_index , distance   
    x = list()
    for j in range(len(distance[0])):
        for i in range(len(distance)):
            x.append( distance[i][j])
        min_dist_index.append( x.index(min (x)))
        x = list()
    distance = list()
 
def get_dist_from_clu_center(c,lists):
    global v_center, distance  
    x =list()
    m = len(lists)
    n = len(lists[0])
    temp = 0.0
    for i in range(m):
        for j in range(n):
            for k in range(m):
                temp = temp + ((lists[k][j] - v_center[i][k])**2)
            x.append(math.sqrt(temp))
            temp = 0.0
        distance.append(x)
        x = list()
   
def get_cluster_centers(c,lists):
    global v_center,vector 
    v_center = list()
    center = list()
    m = len(lists)
    n = len(lists[0])
    temp = 0.0
    denominator = 0.0
    for i in vector:
        for k in lists:
            for l in range(n):
                temp = temp + i[l]* k[l]
                denominator = denominator + i[l]
            if denominator == 0:
                denominator = 1
            center.append(float(temp)/denominator)
            temp = 0.0
            denominator = 0.0
        v_center.append(center)
        center = list()
    
def create_initialvector(c,lists):
    global vector
    for i in range (c):
        vector.append([])
        for j in range(len(lists[0])):
            vector[i].append(0)
    x = 0
    for i in range(len(lists[0])):
        vector[x][i] =1
        x = x+1
        if x == c:
            x= 0
   
def assign_inputs_to_start_nodes():
    x = [list(j) for j in zip(start_nodes,[i for i in range(len(training))]*len(start_nodes))]
    for i in range(len(x)):
        parameter_assigned[x[i][0]] = x[i][1]
  
def assign_outputs_to_end_nodes():
    x =[ list(j) for j in zip (end_nodes,[i for i in range(len(member[0]))])]
    for i in range(len(x)):
        end_member_check[x[i][0]] = x[i][1]

def main():
    accept()
    create_cluster(cluster_count,training)
    assign_inputs_to_start_nodes()
    assign_outputs_to_end_nodes()
    for i in range(len(training[0])):
        iterations = 0
        pass_values_to_network(i)
        while is_err_not_minimum():
            iterations = iterations +1
            pass_values_to_network(i)
            total_error(i)
            error_at_node()          
            update_all_path_weights()
        temp = []
        for i in end_nodes:
            temp.append(node_values[i])
        member.append(temp)
        reset_node_values()
        reset_error_values()
    print "\nPath weights after training"
    for i in path_weights:
        print i+"  "+str(path_weights[i])
    for i in range(len(training[0])):
        pass_check_values_to_network(i)
        while is_err_not_minimum():
            iterations = iterations +1
            pass_check_values_to_network(i)
            total_error(i)
            error_at_node()          
            update_all_path_weights()
        temp = []
        for i in end_nodes:
            temp.append(node_values[i])
        member.append(temp)
    del member[:len(training[0])]
    print "\nMembership"
    for i in member:
        print i

if __name__ == "__main__":
    main()
