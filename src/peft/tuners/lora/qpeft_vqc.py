
import torchquantum as tq
import torch.nn as nn
import torch
import numpy as np

# torch quantum
class VQC(tq.QuantumModule):
    """
    A variational quantum circuit (VQC, Variational Ansatz) consists of three parts: 
    (1) tensor product encoder; (2) variational ansatz; (3) measurement. We can 
    create an encoder by passing a list of gates to tq.GeneralEncoder. Each entry 
    in the list contains input_idx, func, and wires. Here, each qubit has a Pauli-Y 
    gate, which can convert the classical input data into the quantum states. Then, 
    we choose the variational ansatz such that each quantum channel is mutually entangled 
    and Pauli-X,Y,Z gates rotated by arbitrary angles. Finally, we perform Pauli-Z 
    measurements on each qubit on each qubit by creating a tq.MeasureAll module 
    and passing tq.PauliZ to it. The measure function will return four expectation 
    values from the qubits. 
    """
    def __init__(self, 
                 n_wires: int = 8,
                 n_qlayers: int = 1):
        super().__init__()
        self.n_wires = n_wires 
        self.n_qlayers = n_qlayers
            
        # Setting up tensor product encoder
        enc_cnt = list()
        for i in range(self.n_wires):
            cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
            enc_cnt.append(cnt)
        self.encoder = tq.GeneralEncoder(enc_cnt)
        # self.encoder = tq.AmplitudeEncoder()
        
        # We create trainable model parameters, which are stored in dict 
        self.params_ry1_dct = tq.QuantumModuleDict()
        self.params_ry2_dct = tq.QuantumModuleDict()
        self.params_crx1_dct = tq.QuantumModuleDict()
        self.params_crx2_dct = tq.QuantumModuleDict()
            
        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                self.params_ry1_dct[str(i + k*self.n_wires)] = tq.RY(has_params=True, trainable=True)
                self.params_crx1_dct[str(i + k*self.n_wires)] = tq.CRZ(has_params=True, trainable=True)
                self.params_ry2_dct[str(i + k*self.n_wires)] = tq.RY(has_params=True, trainable=True)
                self.params_crx2_dct[str(i + k*self.n_wires)] = tq.CRZ(has_params=True, trainable=True)
 
        self.measure = tq.MeasureMultipleTimes([{'wires': range(self.n_wires), 'observables': ['z'] * self.n_wires}])
        # self.measure = tq.MeasureMultipleTimes(
        #     [{'wires': range(self.n_wires), 'observables': ['x'] * self.n_wires},
        #         {'wires': range(self.n_wires), 'observables': ['y'] * self.n_wires},
        #         {'wires': range(self.n_wires), 'observables': ['z'] * self.n_wires}])
        self.dev = tq.QuantumDevice(n_wires=self.n_wires)

    @tq.static_support 
    def forward(self, x: torch.Tensor):
        """
        1. To convert tq QuantumModule to qiskit or run in the static model,
        we need to:
            (1) add @tq.static_support before the forward
            (2) make sure to add
                static=self.static_mode and 
                parent_graph=self.graph
                to all the tqf functions, such as tqf.hadamard below
        """
        q_device = self.dev
        q_device.reset_states(x.shape[0])
        
        for k in range(self.n_qlayers):
            self.encoder(q_device, x)
                
            for i in range(self.n_wires - 1, -1, -1):
                self.params_crx1_dct[str(i + k*self.n_wires)](q_device, wires=[i, (i + 1) % self.n_wires])

            for i in range(self.n_wires):
                self.params_ry1_dct[str(i + k*self.n_wires)](q_device, wires=i)            
            
            for i in [self.n_wires - 1] + list(range(self.n_wires - 1)):
                self.params_crx2_dct[str(i + k*self.n_wires)](q_device, wires=[i, (i - 1) % self.n_wires])

            for i in range(self.n_wires):
                self.params_ry2_dct[str(i + k*self.n_wires)](q_device, wires=i)

        return (self.measure(q_device))
    

class QLP(nn.Module):
    def __init__(self, n_qubits: int = 8, n_qlayers=1):
        super(QLP, self).__init__()
        self.n_qubits = n_qubits
        # Using a classical feed-forward layer
        self.vqc = VQC(n_wires=n_qubits, n_qlayers=n_qlayers)
        
    def forward(self, input_features):
        # print("input_features", input_features.shape)
        new_input_features = input_features.clone()
        if new_input_features.ndim == 3:
            bsz,len,fz = input_features.size()
            input_features = input_features.view(-1, fz)
        else:
            bsz,fz = input_features.size()
 
        q_in = input_features

        quantum_out = self.vqc(q_in) #.to(device)
        if new_input_features.ndim == 3:
            output = quantum_out.view(bsz,len,fz)
        else:
             output = quantum_out.view(bsz,fz)
        return output


class TTOLayer(nn.Module):
    def __init__(self, 
                 inp_modes, 
                 out_modes, 
                 mat_ranks,
                 cores_initializer=nn.init.kaiming_normal_, 
                 cores_regularizer=None, 
                 biases_initializer=torch.zeros, 
                 biases_regularizer=None, 
                 trainable=True, 
                 cpu_variables=False, 
                 scope=None):
        super(TTOLayer, self).__init__()
        
        self.inp_modes = inp_modes
        self.out_modes = out_modes
        self.mat_ranks = mat_ranks
        self.dim = len(inp_modes)
        self.cpu_variables = cpu_variables
        
        # 创建核心张量列表
        self.mat_cores = nn.ParameterList()
        for i in range(self.dim):
            shape = (out_modes[i] * mat_ranks[i + 1], mat_ranks[i] * inp_modes[i])
            core = torch.empty(shape)
            cores_initializer(core)
            self.mat_cores.append(nn.Parameter(core, requires_grad=trainable))
    
    def forward(self, inp):
        # inp形状: [batch_size, prod(inp_modes)]
        batch_size, len, embed_size = inp.shape
        
        # 重塑输入
        out = inp.view(-1, np.prod(self.inp_modes))
        out = out.t()  # 转置，形状变为 [prod(inp_modes), batch_size]
        
        # 进行核心张量的矩阵乘法
        for i in range(self.dim):            
            # 重塑为 [mat_ranks[i] * inp_modes[i], -1]
            out = out.reshape(self.mat_ranks[i] * self.inp_modes[i], -1)
            out = torch.matmul(self.mat_cores[i], out.to(self.mat_cores[i].dtype))
            # 重塑为 [out_modes[i], -1]
            out = out.reshape(self.out_modes[i], -1)
            # 转置
            out = out.t()        
      
        out = out.reshape(batch_size, len, np.prod(self.out_modes))
        return out